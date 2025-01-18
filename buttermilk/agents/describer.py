

from buttermilk import logger
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.lc import LC
from buttermilk.utils.media import download_and_convert


class Describer(LC):
    template: str | None = "describe"

    async def process_job(
        self,
        *,
        job: Job,
        **kwargs,
    ) -> Job:
        # First step, fetch the record if we need to.
        if job.parameters.pop("download_if_necessary", True):
            if not job.record:
                logger.debug(
                    f"Trying to fetch record for job {job.job_id} from job input parameters.",
                )
                job.record = await download_and_convert(**job.inputs)

        # Next step, call a model to describe the object, but only if necessary.
        # We don't run this step if we only have text components.
        if not job.record or not job.record._components:
            logger.debug(
                f"Not invoking agent {self.name} for job {job.job_id}, no media object provided.",
            )
            return job

        # Skip processing if we already have information about the
        # media object (job.record).
        if job.record.alt_text:
            logger.debug(
                f"Not invoking agent {self.name} for job {job.job_id}, alt text / transcript / media captions already exist.",
            )
            return job

        if not job.parameters.pop("describe", True):
            logger.debug(
                f"Not invoking agent {self.name} for job {job.job_id} due to job config.",
            return job
            )

        return await super().process_job(job=job, **kwargs)
