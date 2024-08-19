import datetime
import platform
import psutil
import shortuuid

def make_run_id() -> str:
    # Create a unique identifier for this run
    node_name = platform.uname().node
    username = psutil.Process().username()
    username = str.split(username, "\\")[
        -1
    ]  # get rid of windows domain if present

    # Format the current datetime as an ISO 8601 string with time zone
    #run_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds') + 'Z'

    # The ISO format has too many special characters for a filename, so we'll use a simpler format
    run_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{platform.uname().node}"

    return run_id