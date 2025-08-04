from fastapi import APIRouter
import json
import time
from fastapi import Query, BackgroundTasks, WebSocket, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from utils import tools, Config, TumourData
from models import model
from task import task_oi
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent

def set_data_root_path():
    Config.BASE_PATH = root_dir / "data" / "duke"

router = APIRouter()

@router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # send a JSON object over the WebSocket connection
    Config.Connected_Websocket = websocket
    try:
        while True:
            message = await websocket.receive_text()
            if Config.Updated_Mesh:
                await send_obj_to_frontend(Config.Current_Case_Name)
                Config.Updated_Mesh = False
    except Exception as e:
        print("closed", e)
        Config.Connected_Websocket = None


async def send_obj_to_frontend(patient_id):
    obj_path = tools.get_file_path(patient_id, "obj", "mask.obj")
    file_exists = (obj_path is not None) and obj_path.exists()
    if file_exists:
        with open(obj_path, "rb") as file:
            file_data = file.read()
        if Config.Connected_Websocket is not None:
            await Config.Connected_Websocket.send_bytes(file_data)
            volume_json = {"volume": TumourData.volume}
            await Config.Connected_Websocket.send_text(json.dumps(volume_json))
            print("send mesh to frontend!")
    else:
        if Config.Connected_Websocket is not None:
            await Config.Connected_Websocket.send_text("delete")
            print("send to frontend, delete mesh!")


@router.get('/api/cases')
async def get_cases_name(background_tasks: BackgroundTasks):
    set_data_root_path()
    background_tasks.add_task(tools.save)
    tools.get_metadata()
    case_names = tools.get_all_case_names()
    case_names.sort()
    res = {}
    res["names"] = case_names
    res["details"] = []
    for name in case_names:
        origin_nrrd_paths = tools.get_category_files(name, "nrrd", "origin")
        registration_nrrd_paths = tools.get_category_files(name, "nrrd", "registration")
        segmentation_breast_points_paths = tools.get_category_files(name, "json", "segmentation")
        segmentation_breast_model_paths = tools.get_category_files(name, "obj", "segmentation")
        # get all masks json files
        segmentation_manual_mask_paths = tools.get_category_files(name, "json", "segmentation_manual",
                                                                  ["sphere_points.json", "tumour_position_study.json",
                                                                   "tumour_position_study_assisted.json"])
        segmentation_manual_3dobj_paths = tools.get_category_files(name, "obj", "segmentation_manual")
        json_is_exist = tools.check_file_exist(name, "json", "mask.json")
        obj_is_exist = tools.check_file_exist(name, "obj", "mask.obj")
        reg_is_exist = tools.check_file_exist(name, "nrrd", "r0.nrrd")
        file_paths = {"origin_nrrd_paths": origin_nrrd_paths,
                      "registration_nrrd_paths": registration_nrrd_paths,
                      "segmentation_breast_points_paths": segmentation_breast_points_paths,
                      "segmentation_breast_model_paths": segmentation_breast_model_paths,
                      "segmentation_manual_mask_paths": segmentation_manual_mask_paths,
                      "segmentation_manual_3dobj_paths": segmentation_manual_3dobj_paths}

        res["details"].append(
            {"name": name, "masked": json_is_exist, "has_mesh": obj_is_exist, "registered": reg_is_exist,
             "file_paths": file_paths})

    return res


@router.get('/api/case/')
async def send_nrrd_case(name: str = Query(None)):
    start_time = time.time()
    if name is not None:
        #  set default images to registration
        tools.zipNrrdFiles(name, "registration")
    end_time = time.time()
    run_time = end_time - start_time
    print("get files cost：{:.2f}s".format(run_time))
    return FileResponse('nrrd_files.zip', media_type='application/zip')


async def process_file(file_path: Path, headers: dict):
    if file_path.suffix == '.nrrd':
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_path.name, headers=headers)
    elif file_path.suffix == '.json':
        file_object = tools.getReturnedJsonFormat(file_path)
        return StreamingResponse(file_object, media_type="application/json", headers=headers)
    elif file_path.suffix == '.obj':
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_path.name, headers=headers)
    else:
        return None


@router.get('/api/single-file')
async def send_single_file(path: str = Query(None)):
    file_path = Path(path)
    if file_path.exists():
        headers = {"x-file-name": file_path.name}
        response = await process_file(file_path, headers)
        if response:
            return response
        else:
            return "Unsupported file format!"
    else:
        return "No file exists!"


@router.get('/api/caseorigin/')
async def send_nrrd_case(name: str = Query(None)):
    if name is not None:
        tools.zipNrrdFiles(name, "origin")
    return FileResponse('nrrd_files.zip', media_type='application/zip')


@router.get('/api/casereg/')
async def send_nrrd_case(data: str):
    data_Obj = json.loads(data)
    name = data_Obj["name"]
    radius = data_Obj["radius"]
    origin = data_Obj["origin"]
    if name is not None:
        tools.zipNrrdFiles(name, "registration")
    return FileResponse('nrrd_files.zip', media_type='application/zip')


@router.post("/api/mask/init")
async def init_mask(mask: model.Masks):
    Config.MASKS = None
    tools.write_data_to_json(mask.caseId, mask.masks)
    return True


@router.post("/api/mask/replace")
async def replace_mask(replace_slice: model.Mask):
    Config.ClearAllMask = False
    tools.replace_data_to_json(replace_slice.caseId, replace_slice)
    return True


@router.post("/api/sphere/save")
async def save_sphere(sphere_point: model.Sphere):
    save_data = {
        "caseId": sphere_point.caseId,
        "sliceId": sphere_point.sliceId,
        "origin": sphere_point.origin,
        "spacing": sphere_point.spacing,
        "sphereRadiusMM": sphere_point.sphereRadiusMM,
        "sphereOriginMM": sphere_point.sphereOriginMM
    }
    return tools.save_sphere_points_to_json(sphere_point.caseId, save_data)


@router.get("/api/mask/save")
async def save_mask(name: str, background_tasks: BackgroundTasks):
    Config.Current_Case_Name = name
    background_tasks.add_task(task_oi.json_to_nii, name)
    background_tasks.add_task(task_oi.on_complete)
    return True


@router.get("/api/mask")
async def get_mask(name: str = Query(None)):
    if name is not None:
        Config.Current_Case_Name = name
        mask_json_path = tools.get_file_path(name, "json", "mask.json")
        checked = tools.check_file_exist(name, "json", "mask.json")
        if checked:
            file_object = tools.getReturnedJsonFormat(mask_json_path)
            return StreamingResponse(file_object, media_type="application/json")
        else:
            return False


@router.get("/api/breast_points")
async def get_breast_points(name: str = Query(None), filename: str = Query(None)):
    checked = tools.check_file_exist(name, "json", f"{filename}.json")
    if checked:
        path = tools.get_file_path(name, "json", f"{filename}.json")
        if "nipple" in filename:
            file_object = tools.getReturnedJsonFormat(path)
            return StreamingResponse(file_object, media_type="application/json")
        else:
            # file_object = tools.getReturnedJsonFormat(path)
            return FileResponse(path, media_type="application/json")
    else:
        return False


@router.get("/api/display")
async def get_display_mask_nrrd(name: str = Query(None)):
    mask_nrrd_path = tools.get_file_path(name, "nrrd", "contrast_0.nrrd")
    if mask_nrrd_path.exists():
        return FileResponse(mask_nrrd_path, media_type="application/octet-stream", filename="mask.nrrd")
    else:
        return False


@router.get("/api/mask_tumour_mesh")
async def get_display_segment_tumour_model(name: str = Query(None)):
    mask_mesh_path = tools.get_file_path(name, "obj", "mask.obj")
    mask_json_path = tools.get_file_path(name, "json", "mask.json")
    if (mask_mesh_path is not None) and (mask_mesh_path.exists()) and (mask_json_path.stat().st_size != 0) and (
            mask_mesh_path.stat().st_size != 0):
        with open(mask_json_path) as user_file:
            file_contents = user_file.read()
            parsed_json = json.loads(file_contents)
            volume = parsed_json["volume"]
            user_file.close()
        mesh_volume_str = json.dumps({"volume": volume})
        headers = {"x-volume": mesh_volume_str}
        file_res = FileResponse(mask_mesh_path, media_type="application/octet-stream", filename="mask.obj",
                                headers=headers)
        return file_res
    else:
        # return False
        raise HTTPException(status_code=404, detail="Item not found")


@router.get("/api/breast_model")
async def get_display_breast_model(name: str = Query(None)):
    breast_mesh_path = tools.get_file_path(name, "obj", "prone_surface.obj")
    if breast_mesh_path is not None and breast_mesh_path.exists():
        file_res = FileResponse(breast_mesh_path, media_type="application/octet-stream", filename="prone_surface.obj")
        return file_res
    else:
        return False


@router.get("/api/clearmesh")
async def clear_mesh(name: str = Query(None)):
    Config.ClearAllMask = True
    mesh_obj_path = tools.get_file_path(name, "obj", "mask.obj")
    if (mesh_obj_path is not None) and mesh_obj_path.exists():
        try:
            mesh_obj_path.unlink()
            print(f"{mesh_obj_path.name} file delete successfully!")
        except OSError as e:
            print(f"fail to delete file!")
    Config.ClearAllMask = False
    return "success"


@router.post("/api/save_tumour_position")
async def save_tumour_position(save_position: model.TumourPosition):
    tumour_position_path = tools.get_file_path(save_position.case_name, "json", "tumour_window.json")
    position_json = {}
    if tumour_position_path.exists():
        with open(tumour_position_path, "r") as tumour_position_file:
            data = tumour_position_file.read()
            position_json = json.loads(data)
    position_json["center"] = save_position.model_dump().get("position")
    position_json["validate"] = save_position.model_dump().get("validate", False)
    with open(tumour_position_path, "w") as tumour_position_file:
        json.dump(position_json, tumour_position_file, indent=4)

    return True