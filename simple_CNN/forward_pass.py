import cv2
import torch
import torch.nn as nn
import numpy as np
from CNN import CNN
# %%
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load model
class_names = ["Accident", "sideswipe", "head-on", "t-bone", "single", "rear-end"]
accident_probabilities = []
model = CNN(len(class_names)).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

# %%
# Prediction function (replacement for predict_accident)
def predict_accident(image):
    """
    image: numpy array (H, W, C)
    """
    img = image.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dim

    tensor = torch.tensor(img).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        prob, pred = torch.max(probs, 1)

    return class_names[pred.item()], probs.cpu().numpy()

# %%
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video = cv2.VideoCapture('../sim_dataset/videos/sideswipe/Town04_sideswipe_clear_00.mp4')
    # For webcam: cv2.VideoCapture(0)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count}, FPS: {fps}")

    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video or error occurred.")
            break

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input
        roi = cv2.resize(rgb_frame, (250, 250))

        pred, probs = predict_accident(roi)
        accident_probabilities.append(probs)
        if pred == "Accident":
            prob = round(probs[0][0] * 100, 2)


            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob}%", (20, 30),
                        font, 1, (255, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# %%
if __name__ == '__main__':
    startapplication()