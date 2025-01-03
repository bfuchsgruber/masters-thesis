
from typing import Literal, List
from datetime import datetime

def tprint(msg="", 
           formatter="",
           date=True,
           datestring="%H:%M:%S.%f",
           ) -> None:
    
    if formatter == "":
        print(msg)
        return None

    retVal = f"[ {formatter} ] {msg}"

    if date:
        ct = datetime.now().strftime(datestring)
        retVal = f"{ct} {retVal}"

    print(retVal)
    
    pass
    return None
