"""
DataCrunch AI Renamer
app.py

Entry point.
"""

from pathlib import Path
from loguru import logger
from models.file_understanding import FileUnderstanding
from utils.safe_move import safe_move
from config import settings
from models.run_statistics import RunStatistics
from services.discovery import discover_files
from services.evaluator import evaluate_prediction
from services.inference import (
    analyze_image,
    analyze_text,
)
from utils.cleanup import cleanup_temp_file
from services.naming import build_filename, validate_filename
from services.preprocess import preprocess
from utils.hashing import sha256_file


def create_directories() -> None:

    directories = [
        settings.DATA_DIR,
        settings.INCOMING_DIR,
        settings.RENAMED_DIR,
        settings.FAILED_DIR,
        settings.DATASET_DIR,
        Path("logs"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def configure_logger():

    logger.remove()

    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    logger.add(
        "logs/app.log",
        level=settings.LOG_LEVEL,
        rotation="5 MB",
        retention="10 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )


def startup():

    configure_logger()

    create_directories()

    logger.info("=" * 60)
    logger.info("DataCrunch AI Renamer")
    logger.info("=" * 60)

    logger.info(f"AI Engine : {settings.AI_ENGINE}")
    logger.info(f"Model     : {settings.MODEL_NAME}")
    logger.info(f"Dataset   : {settings.DATASET_DIR}")
    logger.info(f"Incoming  : {settings.INCOMING_DIR}")
    logger.info("Startup completed successfully.")


if __name__ == "__main__":

    startup()

    jobs = discover_files(settings.INCOMING_DIR)

    logger.info(f"Discovered {len(jobs)} supported files")
    logger.info(f"Host    : {settings.AI_HOST}")
    logger.info(f"Model   : {settings.MODEL_NAME}")

    stats = RunStatistics()

    result_cache: dict[str, FileUnderstanding] = {}

    for job in jobs:

        logger.info("=" * 70)
        logger.info(f"Processing : {job.filename}")

        try:

            file_hash = sha256_file(job.path)
            cached_result = result_cache.get(file_hash)

            if cached_result is not None:

                logger.info(
                    f"Duplicate content detected (hash match) — "
                    f"reusing prior classification for this run."
                )

                result = cached_result

            else:

             
                # PREPROCESS

                preprocess(job)

                if job.has_errors:
                    raise RuntimeError(job.errors)

             
                # AI UNDERSTANDING

                if job.is_text_document:

                    result = analyze_text(
                        job.extracted_text
                    )

                else:

                    if job.rendered_image_path is None:

                        raise RuntimeError(
                            "Rendered image was not created."
                        )

                    result = analyze_image(
                        job.rendered_image_path
                    )

                result_cache[file_hash] = result

            logger.info(
                result.model_dump_json(indent=2)
            )

            
            stats.processed += 1
            stats.total_confidence += result.confidence

  
            # CONFIDENCE


            is_security_flag = (
                result.category.value == "other"
                and result.confidence <= 0.2
            )

            if is_security_flag or result.confidence < settings.CONFIDENCE_THRESHOLD:

                if is_security_flag:
                    logger.warning(
                        f"SECURITY: '{job.filename}' flagged as suspicious "
                        f"(possible injection attempt in file content) — quarantined."
                    )
                else:
                    logger.warning(
                        f"Low confidence ({result.confidence:.2f}) — {job.filename}"
                    )

                failed = settings.FAILED_DIR / job.filename
                safe_move(job.path, failed)

                job.path.rename(failed)

                stats.failed += 1

                continue
           
            # BUILD FILENAME

            new_name = build_filename(
                result,
                job.path,
            )

            if not validate_filename(new_name):
                raise ValueError(
                    f"Invalid filename: {new_name}"
                )


            # MOVE FILE
            destination = settings.RENAMED_DIR / new_name
            destination = safe_move(job.path, destination)

            stats.renamed += 1

            logger.success(
                f"{job.filename} -> {destination.relative_to(settings.RENAMED_DIR)}"
            )


            #  GOLDEN DATASET EVALUATION


            if settings.DATASET_DIR in job.path.parents:

                evaluation = evaluate_prediction(
                    result,
                    job.path,
                )

                logger.info(
                    f"Expected  : {evaluation['expected']}"
                )

                logger.info(
                    f"Predicted : {evaluation['predicted']}"
                )

                logger.info(
                    f"Confidence: {evaluation['confidence']:.2f}"
                )

                if evaluation["correct"]:

                    stats.correct += 1

                    logger.success(
                        "Prediction Correct"
                    )

                else:

                    stats.incorrect += 1

                    logger.warning(
                        "Prediction Incorrect"
                    )

        except Exception as e:

            stats.failed += 1

            logger.exception(e)

            try:

                if job.path.exists():

                    failed = settings.FAILED_DIR / job.filename
                    safe_move(job.path, failed)

            except Exception:

                logger.exception(
                    "Unable to move file."
                )
        finally:

            cleanup_temp_file(job.rendered_image_path)



    # SUMMARY

    logger.info("=" * 70)
    logger.info("RUN SUMMARY")
    logger.info("=" * 70)

    logger.info(f"Files Found        : {len(jobs)}")
    logger.info(f"Processed          : {stats.processed}")
    logger.info(f"Renamed            : {stats.renamed}")
    logger.info(f"Failed             : {stats.failed}")
    logger.info(f"Correct            : {stats.correct}")
    logger.info(f"Incorrect          : {stats.incorrect}")
    logger.info(
        f"Average Confidence : {stats.average_confidence:.2f}"
    )

    logger.success("Pipeline Finished.")