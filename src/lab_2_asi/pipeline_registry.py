from kedro.framework.project import find_pipelines

def register_pipelines():
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines