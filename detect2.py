import argparse
import time
from pathlib import Path

import base64
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import matplotlib.pyplot as plt
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# import torch
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
# from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


origins = [
    "http://localhost",
    "http://localhost:9999",
]
middleware = [
    Middleware(CORSMiddleware, allow_origins=origins)
]

app = FastAPI(middleware=middleware)
# app.add_middleware(GZipMiddleware)
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])
app.mount("/static", StaticFiles(directory="static"), name="static")
def detect(name,save_img=False):
    # source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    
    weights = '/kaggle/best.pt'
    imgsz = 512
    save_txt = False
    source ='/kaggle/a/'+name
    print(source)
    view_img=True   

    total_e = ''

    exist_ok=True
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    # save_dir = Path(increment_path(Path('/detect') / 'exp', exist_ok=True))  # increment run
    save_dir = Path(increment_path(Path('static') / 'exp', exist_ok=True)) 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.15, 0.5, classes=None, agnostic=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        total_e += f'{names[int(cls)]} {conf:.2f},'
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            print(s)
            result = s
            # Stream results
            
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     if cv2.waitKey(0) == ord('q'):  # q to quit
            #         raise StopIteration

            # Save results (image with detections)
            print(save_path)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    # data64 = base64.b64encode(im0)
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(total_e)
    return {
            "result":total_e,
            "path":  save_path
            }

@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.post("/inference")
# def inference_with_path(imgs: Image):
#     return results_to_json(model(imgs.img_list))
opt = []
@app.get("/test")
def read_root():
    # source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # opt = parser.parse_args()
    print(opt)
    
    return {"result": detect()}


@app.post("/files/")
async def create_file(files: bytes = File(...)):
    return {"file_size": len(files)}

# @app.post("/files/")
# async def create_file(
#     files: bytes = File(...), fileb: UploadFile = File(...), token: str = Form(...)
# ):
#     return {
#         "file_size": len(file),
#         "token": token,
#         "fileb_content_type": fileb.content_type,
#     }

@app.post("/uploadfiles/")
async def create_upload_file(files: UploadFile = File(...)):
    image = await files.read();
    # print(image)
    print(files.file)
    
    file_location = f"/kaggle/a/{files.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(image)
    print({"info": f"file '{files.filename}' saved at '{file_location}'"})

    # return {"result": detect(name = files.filename)} 
    data = detect(name = files.filename)
    content = f"""
<body>
    <h1>{data['result']}</h1>
    <img src="http://192.168.3.133:9999/{data['path']}" alt="Girl in a jacket" width="500" height="600">
</body>
    """
    return HTMLResponse(content=content)

@app.post("/uploadfilesjs/")
async def create_upload_file(files: UploadFile = File(...)):
    image = await files.read();
    # print(image)
    print(files.file)
    
    file_location = f"/kaggle/a/{files.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(image)
    print({"info": f"file '{files.filename}' saved at '{file_location}'"})

    return {"result": detect(name = files.filename)} 
    # data = detect(name = files.filename)
    # return 
@app.post("/predict/")
async def predict(files: UploadFile = File(...)):
    extension = files.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await files.read())
    cv2.imshow("sad",image)

    return "have"


@app.get("/haha")
async def web():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file">
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    save_text = parser.parse_args().save_txt
    app_str = 'detect2:app'
    uvicorn.run(app_str, host='localhost', port=9999, log_level='info', reload=True, workers=1,debug=True)
    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()
#  uvicorn detect2:app --port 9999 --host 0.0.0.0