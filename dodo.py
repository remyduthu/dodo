#!usr/bin/env python3

import cv2
import mediapipe

pose = mediapipe.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)


def openSample(fileName: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(fileName)


def processSample(sample: cv2.VideoCapture) -> None:
    while sample.isOpened():
        hasNext, frame = sample.read()
        if not hasNext:
            break

        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            continue

        mediapipe.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mediapipe.solutions.pose.POSE_CONNECTIONS
        )

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)


def closeSample(sample: cv2.VideoCapture) -> None:
    sample.release()

    cv2.destroyAllWindows()


def main():
    samples = [
        openSample("./samples/0.mov"),
        openSample("./samples/1.mov"),
        openSample("./samples/2.mov"),
    ]
    for sample in samples:
        processSample(sample)
        closeSample(sample)


if __name__ == "__main__":
    main()
