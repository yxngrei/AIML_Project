import cv2
import numpy as np

# Filter modes
PREVIEW = 0       # Preview mode
BLUR = 1          # Blur mode
FEATURES = 2      # Features mode
CANNY = 3         # Canny mode
GRAYSCALE = 4     # Grayscale mode
LAPLACIAN = 5     # Laplacian edge detection mode
THRESHOLD = 6     # Threshold mode
BILATERAL = 7     # Bilateral filtering mode

# Parameters for feature detection
features_params = dict(maxCorners=500, qualityLevel=0.1, minDistance=15, blockSize=9)

s = 0  # for default camera

# Set default filter mode to preview
image_filter = PREVIEW

source = cv2.VideoCapture(s)

alive = True
window_name = "Camera Filters"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
result = None

mode_names = {
    PREVIEW: "Preview",
    BLUR: "Blur",
    FEATURES: "Features",
    CANNY: "Canny",
    GRAYSCALE: "Grayscale",
    LAPLACIAN: "Laplacian",
    THRESHOLD: "Threshold",
    BILATERAL: "Bilateral"
}

while alive:
    has_frame, frame = source.read()

    if not has_frame:
        break

    frame = cv2.flip(frame, 1)

    try:
        if image_filter == PREVIEW:
            result = frame

        elif image_filter == BLUR:
            result = cv2.GaussianBlur(frame, (21, 21), 0)

        elif image_filter == CANNY:
            result = cv2.Canny(frame, 30, 200)

        elif image_filter == FEATURES:
            result = frame
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(frame_gray, **features_params)
            if corners is not None:
                corners = np.intp(corners)
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(result, (x, y), 10, (0, 255, 0), 1)

        elif image_filter == GRAYSCALE:
            result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        elif image_filter == LAPLACIAN:
            result = cv2.Laplacian(frame, cv2.CV_64F)

        elif image_filter == THRESHOLD:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        elif image_filter == BILATERAL:
            result = cv2.bilateralFilter(frame, 9, 75, 75)

        mode_text = mode_names[image_filter]
        cv2.putText(result, mode_text, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, result)

        key = cv2.waitKey(1)
        if key == ord("Q") or key == ord("q") or key == 27:
            alive = False  
        elif key == ord("C") or key == ord("c"):
            image_filter = CANNY  
        elif key == ord("B") or key == ord("b"):
            image_filter = BLUR   
        elif key == ord("F") or key == ord("f"):
            image_filter = FEATURES  
        elif key == ord("P") or key == ord("p"):
            image_filter = PREVIEW  
        elif key == ord("G") or key == ord("g"):
            image_filter = GRAYSCALE  
        elif key == ord("L") or key == ord("l"):
            image_filter = LAPLACIAN  
        elif key == ord("T") or key == ord("t"):
            image_filter = THRESHOLD  
        elif key == ord("D") or key == ord("d"):
            image_filter = BILATERAL  

    except Exception as e:
        print("An error occurred:", str(e))
        break

source.release()
cv2.destroyWindow(window_name)