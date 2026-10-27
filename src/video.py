import cv2

cap = cv2.VideoCapture(0)   # Enter filename instead of index

if not cap.isOpened():
    print("Couldn't open the video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter("results/output.mp4", fourcc=fourcc, fps=fps, frameSize=(frame_width, frame_height), isColor=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error occured.")
        break

    out.write(frame)

    cv2.imshow("Frame:", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()