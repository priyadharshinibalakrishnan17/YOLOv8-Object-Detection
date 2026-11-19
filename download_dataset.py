
from roboflow import Roboflow
rf = Roboflow(api_key="zEh1eam7H6y8pBKqXigo")
project = rf.workspace("kowsiks-workspace-yvwjl").project("team-alpha-b8bpo-i1m7c")
version = project.version(2)
dataset = version.download("yolov8")
