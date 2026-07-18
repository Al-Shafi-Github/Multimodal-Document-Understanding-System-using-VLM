from dataclasses import dataclass


@dataclass
class RunStatistics:
    """
    Statistics for one execution of the pipeline.
    """

    # Files discovered
    total_files: int = 0

    # Successfully processed
    processed: int = 0

    # Successfully renamed
    renamed: int = 0

    # Failed
    failed: int = 0

    # Evaluation
    correct: int = 0
    incorrect: int = 0

    # Retry count
    retries: int = 0

    # Confidence accumulation
    total_confidence: float = 0.0

    @property
    def average_confidence(self) -> float:

        if self.processed == 0:
            return 0.0

        return self.total_confidence / self.processed

    @property
    def accuracy(self) -> float:

        total = self.correct + self.incorrect

        if total == 0:
            return 0.0

        return self.correct / total