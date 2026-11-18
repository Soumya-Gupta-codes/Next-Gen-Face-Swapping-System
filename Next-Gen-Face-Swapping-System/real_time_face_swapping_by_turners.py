import os
import sys
import argparse
import time
import cv2
import numpy as np
import dlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def main():
    parser = argparse.ArgumentParser(description="Next-Gen Face Swapping System using OpenCV and Dlib")
    parser.add_argument("--source", default=os.path.join(BASE_DIR, "assets", "source_face.jpg"),
                        help="Path to source face image (default: assets/source_face.jpg)")
    parser.add_argument("--target", default=None,
                        help="Path to target image or video file (default: webcam stream)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat"),
                        help="Path to shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--headless", action="store_true",
                        help="Run without displaying interactive GUI windows (useful for headless testing or scripting)")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "assets", "output_swapped.jpg"),
                        help="Output path for swapped image when target is an image or in headless mode")
    args = parser.parse_args()

    # Verify model exists
    if not os.path.exists(args.model):
        print(f"Error: Landmark predictor model not found at '{args.model}'.")
        print("Please run 'python download_model.py' to download the model.")
        sys.exit(1)

    # Verify source image exists
    if not os.path.exists(args.source):
        print(f"Error: Source image not found at '{args.source}'.")
        sys.exit(1)

    img = cv2.imread(args.source)
    if img is None:
        print(f"Error: Failed to read source image '{args.source}'.")
        sys.exit(1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model)

    indexes_triangles = []

    # Face 1
    faces = detector(img_gray)
    if len(faces) == 0:
        print(f"Error: No face detected in source image '{args.source}'. Please provide an image with a clearly visible face.")
        sys.exit(1)

    landmarks = predictor(img_gray, faces[0])
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)

    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    # Determine input mode (static image vs video / webcam)
    is_static_image = False
    cap = None
    if args.target:
        if not os.path.exists(args.target):
            print(f"Error: Target path '{args.target}' does not exist.")
            sys.exit(1)
        # Check if target is an image
        test_img = cv2.imread(args.target)
        if test_img is not None:
            is_static_image = True
            static_target_img = test_img
        else:
            cap = cv2.VideoCapture(args.target)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Warning: Could not open camera {args.camera}.")
            print("Tip: If you do not have a working webcam connected, you can specify a target image using:")
            print("     python real_time_face_swapping_by_turners.py --target assets/target_face.jpg")
            sys.exit(1)

    print("Starting Next-Gen Face Swapping System. Press 'q' in any OpenCV window to exit.")

    while True:
        if is_static_image:
            img2 = static_target_img.copy()
        else:
            ret, img2 = cap.read()
            if not ret or img2 is None:
                print("End of video stream or failed to read frame.")
                break

        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(img2)

        # Face 2
        faces2 = detector(img2_gray)
        if len(faces2) == 0:
            if not args.headless:
                display_frame = img2.copy()
                cv2.putText(display_frame, "No face detected in video stream", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("img2", display_frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            if is_static_image:
                print("No face detected in target image.")
                break
            continue

        for face in faces2:
            landmarks = predictor(img2_gray, face)
            landmarks_points2 = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points2.append((x, y))

            points2 = np.array(landmarks_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)
            break  # Swap first detected face

        lines_space_mask = np.zeros_like(img_gray)
        lines_space_new_face = np.zeros_like(img2)

        # Triangulation of both faces
        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            tr1_pt1 = landmarks_points[triangle_index[0]]
            tr1_pt2 = landmarks_points[triangle_index[1]]
            tr1_pt3 = landmarks_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            if w <= 0 or h <= 0:
                continue

            cropped_triangle = img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)

            pts1 = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                             [tr1_pt2[0] - x, tr1_pt2[1] - y],
                             [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, pts1, 255)

            # Triangulation of second face
            tr2_pt1 = landmarks_points2[triangle_index[0]]
            tr2_pt2 = landmarks_points2[triangle_index[1]]
            tr2_pt3 = landmarks_points2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2
            if w <= 0 or h <= 0:
                continue

            cropped_tr2_mask = np.zeros((h, w), np.uint8)

            pts2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                             [tr2_pt2[0] - x, tr2_pt2[1] - y],
                             [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, pts2, 255)

            # Warp triangles
            pts1_f = np.float32(pts1)
            pts2_f = np.float32(pts2)
            M = cv2.getAffineTransform(pts1_f, pts2_f)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing destination face
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

        # Face swapped (putting 1st face into 2nd face)
        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        try:
            seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
        except cv2.error:
            seamlessclone = result

        if args.headless or is_static_image:
            cv2.imwrite(args.output, seamlessclone)
            print(f"Swapped output saved to '{args.output}'.")

        if not args.headless:
            cv2.imshow("img2", img2)
            cv2.imshow("clone", seamlessclone)
            cv2.imshow("result", result)

            key = cv2.waitKey(0 if is_static_image else 1)
            if key == ord("q") or is_static_image:
                break
        else:
            break

    if cap is not None:
        cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
