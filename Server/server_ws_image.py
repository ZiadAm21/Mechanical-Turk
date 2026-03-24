#!/usr/bin/env python3
# server_ws_images.py
# aiohttp WebSocket server that accepts binary images and text commands,
# and can broadcast command strings to all connected clients.
    
import asyncio, socket
from aiohttp import web, WSMsgType
from torchvision.transforms import transforms
import logging, os, io
from PIL import Image, ImageEnhance
from typing import Set
from ultralytics import YOLO
import random
from math import sqrt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.basename(os.path.dirname(SCRIPT_DIR))

# Set up for predictions
model = YOLO(os.path.join(SCRIPT_DIR, "AI-Model/best.pt"))
class_dict = {
    0:'bishop', 1:'black-bishop', 2:'black-king', 3:'black-knight', 4:'black-pawn', 
    5:'black-queen', 6:'black-rook', 7:'white-bishop', 8:'white-king', 9:'white-knight', 
    10:'white-pawn', 11:'white-queen', 12:'white-rook'}
resize = transforms.Compose(
                [ transforms.Resize((640,640)), transforms.ToTensor()])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Print local IP address
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print(f"Server starting on IP: {local_ip}:8080")

current_coordinates = "A1"  # starting at A1

TOTAL_STEPS = 400  # total steps to get from one side of chess board to opposite side
UNIT_STEP = 50

def calculate_movement(current, target):
    if not target:
        return "s", "s", 0  # no movement
    elif not "A" <= target[0] >= "H":
        return "s", "s", 0  # invalid target
    elif not "1" <= target[1] <= "8":
        return "s", "s", 0  # invalid target

    files = ["A","B","C","D","E","F","G","H"]
    ranks = ["1","2","3","4","5","6","7","8"]

    file_diff = abs(files.index(current[0]) - files.index(target[0]))
    rank_diff = abs(ranks.index(current[1]) - ranks.index(target[1]))

    file_steps = file_diff * UNIT_STEP
    rank_steps = rank_diff * UNIT_STEP

    steps = sqrt(file_steps**2 + rank_steps**2)  # Pythagorean theorem

    dir1 = "s"
    dir2 = "s"
    if file_steps > 0:
        dir1 = "f"
    elif file_steps < 0:
        dir1 = "b"
        
    if rank_steps > 0:
        dir2 = "f"
    elif rank_steps < 0:
        dir2 = "b"
    return dir1, dir2, steps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws-server")

CONNECTED: Set[web.WebSocketResponse] = set()

# model = YOLO("./yolov8n.pt")

# helpers to format compact commands
def make_stepper_string(dir1: int, dir2: int, steps: int, emag: int) -> str:
    logger.info(f"stepper{dir1}{dir2}{emag}{steps}")
    return f"stepper{dir1}{dir2}{emag}{steps}"

# public API to broadcast commands
async def broadcast(text: str):
    # logger.info("Broadcasting:", text)
    if not CONNECTED:
        logger.info("No clients connected - not broadcasting: %s", text)
        return
    logger.info("Broadcasting -> %s", text)
    to_remove = []
    for ws in list(CONNECTED):
        try:
            await ws.send_str(text)
        except Exception as e:
            logger.warning("Failed to send to client: %s", e)
            to_remove.append(ws)
    for ws in to_remove:
        CONNECTED.discard(ws)

async def send_stepper_command(dir1: str, dir2: str, steps: int, emag: int) -> None:
    if isinstance(steps, float) and not steps.is_integer():
        steps = ("{:.2f}".format(steps)).rstrip('0').rstrip('.')
    else:
        steps = str(int(steps))
    cmd = make_stepper_string(dir1, dir2, steps, emag)
    await broadcast(cmd)

async def move_piece(current, target_piece, target_square):
    dir1, dir2, steps = calculate_movement(current, target_piece)
    await send_stepper_command(dir1, dir2, steps, 0)

    await asyncio.sleep(0.5)

    dir1, dir2, steps = calculate_movement(target_piece, target_square)
    await send_stepper_command(dir1, dir2, steps, 1)

# HTTP helpers (for quick testing from browser/curl)
async def http_send_stepper(request):
    # /send_stepper/{target_piece}/{target_square}/
    target_piece = request.match_info.get("target_piece", "")
    target_square = request.match_info.get("target_square", "")

    await move_piece(current_coordinates, target_piece, target_square)
    return web.json_response({"ok": True, "moved": f"{target_piece} to {target_square}"})

# WebSocket handler
direction = 1
num_image = 0
async def websocket_handler(request):
    global current_coordinates
    global direction
    global num_image
    char_dir = "f"
    ws = web.WebSocketResponse(heartbeat=15.0)  # heartbeat helps keep NAT alive
    await ws.prepare(request)

    CONNECTED.add(ws)
    logger.info("Client connected (%d clients)", len(CONNECTED))

    # optionally send a small hello
    await ws.send_str("hello")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                txt = msg.data
                logger.info("Text msg from client: %s", repr(txt))
                # You may want to respond/ack here
            elif msg.type == WSMsgType.BINARY:
                # Assume JPEG bytes; save to file (rotate or timestamp for multiple)
                data = msg.data
                try:
                    img = Image.open(io.BytesIO(bytes(data))).convert("RGB")
                    enhancer = ImageEnhance.Brightness(img)
                    # img = enhancer.enhance(1.7)
                    predictions = model.predict(img)[0]
                    if len(predictions.boxes) > 0:
                        num_image += 1
                        save_path = os.path.join(SCRIPT_DIR, f"detections\\{num_image}.jpg")
                        predictions.save(save_path)
                        print(f"Saved {save_path} with {len(predictions.boxes)} detections")
                    if random.random() > 0.3:
                        target_piece = random.choice(["A","B","C","D","E","F","G","H"]) + str(random.randint(1,8))
                        target_square = random.choice(["A","B","C","D","E","F","G","H"]) + str(random.randint(1,8))
                        
                        target_piece = "A2"
                        target_square = "E6"

                        await move_piece(current_coordinates, target_piece, target_square)
                        current_coordinates = target_square

                except Exception as e:
                    logger.exception("Failed to save image: %s", e)
            elif msg.type == WSMsgType.ERROR:
                logger.error("WS connection error: %s", ws.exception())
    except Exception as e:
        logger.exception("WS handler exception: %s", e)
    finally:
        CONNECTED.discard(ws)
        logger.info("Client disconnected (%d clients)", len(CONNECTED))
        await ws.close()
    return ws


# basic index
async def index(request):
    return web.Response(text="WebSocket image server. Connect to /ws", content_type="text/plain")

# app setup
app = web.Application()
app.router.add_get("/", index)
app.router.add_get("/ws", websocket_handler)
app.router.add_get("/move/{target_piece}/{target_square}", http_send_stepper)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8080)
