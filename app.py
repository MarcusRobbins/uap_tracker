from aiohttp import web
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

async def get_cameras(request):
    cameras = camera_loop.get_available_cameras()
    return web.json_response(cameras)

async def start_camera(request):
    camera_id = request.match_info.get('camera_id')
    shared_vars.camera_enabled = True
    camera_loop.start_stream(camera_id)
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
        await response.write(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

    return response

app.router.add_get('/cameras', get_cameras)
app.router.add_post('/start_camera/{camera_id}', start_camera)
app.router.add_post('/start_tracking/{camera_id}', start_tracking)
app.router.add_get('/stream/{camera_id}', stream_camera)

web.run_app(app)
