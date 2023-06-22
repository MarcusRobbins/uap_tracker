from aiohttp import web
import aiohttp_cors
import cv2
from src.shared_variables import SharedVariables
from src.camera_loop import CameraLoop
from src.neural_network import NeuralNetworkProcessing
from src.track_ranking_and_storage import TrackRankingAndStorage
 
# Shared Variables
shared_vars = SharedVariables()

# Camera Loop
camera_loop = CameraLoop(shared_vars)

# Neural Network Processing
neural_net = NeuralNetworkProcessing(shared_vars)

# Filtering Process
track_proc = TrackRankingAndStorage(shared_vars)

# Server API
app = web.Application()

# Configure default CORS settings.
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
})

async def get_cameras(request):
    cameras = camera_loop.get_available_cameras()
    return web.json_response(cameras)

async def start_camera(request):
    camera_id = request.match_info.get('camera_id')
    # convert to integer:
    camera_id = int(camera_id)
    camera_loop.start_camera(camera_id=camera_id)
    shared_vars.camera_enabled = True
    return web.Response(text='Camera Started')

async def start_tracking(request):
    camera_id = request.match_info.get('camera_id')
    shared_vars.tracking_enabled = True
    camera_loop.start_tracking(camera_id)
    return web.Response(text='Tracking Started')

async def stream_camera(request):
    response = web.StreamResponse()
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)

    while shared_vars.camera_enabled:
        frame = shared_vars.frame_queue.get()  # Consume frame from queue

        # Convert frame to JPEG
        _, jpeg_frame = cv2.imencode('.jpg', frame)
        jpeg_frame = jpeg_frame.tobytes()

        await response.write(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n'
        )

    return response

async def handle_index(request):
    with open('web_interface/web.html', 'r') as f:
        response_text = f.read()
    return web.Response(text=response_text, content_type='text/html')

app.router.add_get('/cameras', get_cameras)
app.router.add_post('/start_camera/{camera_id}', start_camera)
app.router.add_post('/start_tracking/{camera_id}', start_tracking)
app.router.add_get('/stream/{camera_id}', stream_camera)
app.router.add_get('/', handle_index)


# Apply CORS to routes
resource = cors.add(app.router.add_resource("/cameras"))
route = cors.add(resource.add_route("GET", get_cameras))


web.run_app(app)
