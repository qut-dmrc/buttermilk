

from buttermilk import logger
from buttermilk._core.runner_types import Job
from buttermilk.agents.lc import LC


class Describer(LC):
    template: str = "describe"

    async def process_job(
        self,
        *,
        job: Job,
        **kwargs,
    ) -> Job:
        # only process if we have a media object
        if not job.record or not job.record._components:
            logger.debug(
                f"Not invoking agent {self.name} for job {job.job_id}, no media object provided.",
            )
            return job

        # Skip processing if we already have information about the
        # media object (job.record).
        if job.record.transcript:
            logger.debug(
                f"Not invoking agent {self.name} for job {job.job_id}, transcript / media captions already exist.",
            )
            return job

        return await super().process_job(job=job, **kwargs)
