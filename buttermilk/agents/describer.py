

from buttermilk import logger
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
            if not job.record and job.inputs:
                logger.debug(
                    f"Trying to fetch record for job {job.job_id} from job input parameters {job.inputs.keys()}.",
                )
                job.record = await download_and_convert(**job.inputs)

        # Next step, call a model to describe the object, but only if necessary.
        # We don't run this step if we only have text components.
        if not job.record or not job.record.components:
            logger.debug(
                f"Not invoking agent {self.name} for job {job.job_id}, no media object provided.",
            )
            return job

        media_exists = False
        for component in job.record.components:
            if (
                component.mime.startswith("image")
                or component.mime.startswith("video")
                or component.mime.startswith("audio")
            ):
                media_exists = True
        if not media_exists:
            # don't try to describe only text components
            logger.debug(
                f"Not invoking agent {self.name} for job {job.job_id}, no non-text components provided.",
            )
            return job

        # Skip processing if we already have information about the
        # media object (job.record).
        if job.record.alt_text:
            logger.debug(
                f"Not invoking agent {self.name} for job {job.job_id}, alt text / transcript / media captions already exist.",
            )
            return job

        result = await super().process_job(job=job, **kwargs)

        # Update record alt text, title, description etc.
        result.record.update_from(result.outputs)

        return result
