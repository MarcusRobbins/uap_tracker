from aiohttp import web
import aiohttp_cors
import cv2
from src.shared_variables import SharedVariables
from src.camera_loop import CameraLoop
from src.neural_network import NeuralNetworkProcessing
from src.track_ranking_and_storage import TrackRankingAndStorage
import random
import time
 
if __name__ == '__main__':
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
        # camera_loop.start_camera(camera_id=camera_id)
        shared_vars.camera_enabled = True
        return web.Response(text='Camera Started')

    async def start_tracking(request):
        camera_id = request.match_info.get('camera_id')
        shared_vars.tracking_enabled = True
        # camera_loop.start_tracking(camera_id)
        return web.Response(text='Tracking Started')

    # 'plot_one_box' is a helper function to draw the detection bounding boxes on the image
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    def draw_fps(image, fps, location=(10, 50), color=(255, 0, 0), thickness=2, fontScale=1):
        """Draw the FPS on an image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(image, fps_text, location, font, fontScale, color, thickness, cv2.LINE_AA)
        return image

    async def stream_camera(request):

        response = web.StreamResponse()
        response.content_type = 'multipart/x-mixed-replace; boundary=frame'
        await response.prepare(request)

        # Record start time for FPS calculation
        start_time = time.time()

        while shared_vars.camera_enabled:
            if not shared_vars.frame_queue.empty() and shared_vars.filtered_frame_queue.empty():
                frame = shared_vars.frame_queue.get()  # Consume frame from queue

                # # If tracking data is available, overlay it on the frame
                # if not shared_vars.tracking_queue.empty():
                #     track_data = shared_vars.tracking_queue.get()

                #     for box, score, label in zip(track_data['boxes'], track_data['scores'], track_data['labels']):
                #         # Convert box coordinates to integers
                #         box = [int(x) for x in box]

                #         # Create label with score
                #         label = f'{label} {score:.2f}'

                #         # Overlay box and label on frame
                #         frame = plot_one_box(box, frame, label=label, color=(255, 0, 0), line_thickness=3)

                # cv2.waitKey(1)
                # fps = 1.0 / (time.time() - start_time)
                # draw_fps(frame, fps)

                # Convert frame to JPEG
                _, jpeg_frame = cv2.imencode('.jpg', frame)
                jpeg_frame = jpeg_frame.tobytes()

                
                # # Record start time for FPS calculation
                # start_time = time.time()

                await response.write(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n'
                )

            if not shared_vars.filtered_frame_queue.empty():
                frame = shared_vars.filtered_frame_queue.get()  # Consume frame from queue

                # cv2.waitKey(1)
                # fps = 1.0 / (time.time() - start_time)
                # draw_fps(frame, fps)

                # Convert frame to JPEG
                _, jpeg_frame = cv2.imencode('.jpg', frame)
                jpeg_frame = jpeg_frame.tobytes()

                
                # Record start time for FPS calculation
                # start_time = time.time()

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
