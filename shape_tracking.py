import cv2
import numpy as np

def image_resize(image, width = None, height = None):
    # Fungsi untuk resize dengan ratio yang sama
    inter = cv2.INTER_AREA

    # ambil lebar dan tinggi gambar asli
    (h, w) = image.shape[:2]

    # jika argumen width dan height None, return gambar asli
    if width is None and height is None:
        return image

    # jika width None tapi height tidak
    if width is None:
        # hitung rasio lalu apply ke width
        r = height / float(h)
        dim = (int(w * r), height)

    # the height is None
    else:
        # hitung rasio lalu apply ke height
        r = width / float(w)
        dim = (width, int(h * r))
    
    # resize gambar
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

def detect_shape(contour):
    """Deteksi bentuk objek berdasarkan contour."""
    # simplifikasi bentuk objek
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    sides = len(approx)

    if sides == 3:
        return "Segitiga"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Persegi" if 0.95 <= aspect_ratio <= 1.05 else "Segi Empat"
    elif sides <= 8:
        return "Poligon"
    elif sides > 8:
        return "Lingkaran"
    else:
        return "Kurang Tau"

# Capture video
cap = cv2.VideoCapture('D:/KaderUro/CV/object_video.mp4')

while True:
    # baca tiap frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame lalu convert HSV
    frame = image_resize(frame, width=1080)
    hsvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masking
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsvimg, lower1, upper1)
    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsvimg, lower2, upper2)
    finalmask = cv2.bitwise_or(mask1, mask2)
    bnw = cv2.bitwise_and(hsvimg, hsvimg, mask=finalmask)

    # Canny edge detection
    edges = cv2.Canny(bnw, 50, 150)
        
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Kalau objek besar
            # Deteksi bentuk
            shape = detect_shape(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Buat bounding box dan tulis bentuknya
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, shape, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            break

    cv2.imshow("Shape Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()