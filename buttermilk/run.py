
# cfg changes with multiple options selected at runtime, but
# we want to make sure that run_id (which informs the logging
# tags and save directories) does not change across processes.
import hydra
from buttermilk.bm import BM, Project
from rich import print as rprint

global_run_id = SessionInfo.make_run_id()

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: Project) -> None:
    global bm, logger, global_run_id
    
    rprint(cfg)
    pass

    # Load the main ButterMilk singleton instance
    # This takes care of credentials, save paths, and other defaults
    # It also holds the project configuration
    bm = BM(cfg=cfg)
    logger = bm.logger

if __name__ == '__main__':
    main()
