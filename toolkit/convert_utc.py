from datetime import datetime
from dateutil import tz

def utc_to_local(utc: datetime) -> datetime:
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    utc = utc.replace(tzinfo=from_zone)
    local = utc.astimezone(to_zone)

    return local