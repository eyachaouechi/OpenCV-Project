import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import random
import time
import pygame

# Initialize pygame and mixer
pygame.init()
pygame.mixer.init()

# Load sound files
score_sound = pygame.mixer.Sound("Resources/score.mp3")

loading_video_path = "Resources/Loading.mp4"
loading_duration = 25
loading_start_time = time.time()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Importing all images
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/Right.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/Right.png", cv2.IMREAD_UNCHANGED)

imgBall = cv2.resize(imgBall, (50, 50))
imgBat1 = cv2.resize(imgBat1, (50, 150))
imgBat2 = cv2.resize(imgBat2, (50, 150))
imgBackground = cv2.resize(imgBackground, (int(cap.get(3)), int(cap.get(4))))

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]
timer_start = time.time()
game_over_time = None
duration = 60

# Leaderboard
leaderboard_file = "leaderboard.txt"
while time.time() - loading_start_time < loading_duration:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    # Resize img with reduced height
    img = cv2.resize(img, (img.shape[1], int(img.shape[0] * 0.5)))
    # Load and display the video
    ret, frame = cap.read()
    if not ret:
        break
    video_cap = cv2.VideoCapture(loading_video_path)

    while True:
        _, video_frame = video_cap.read()
        if video_frame is None:
            # Restart video if it ends
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        video_frame = cv2.resize(video_frame, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(img, 0.5, video_frame, 0.5, 0)
        loading_text = "Are You ready !"
        text_size, _ = cv2.getTextSize(loading_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 3
        cv2.putText(img, loading_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Loading", img)
        key = cv2.waitKey(25) & 0xFF
        if key == 27 or time.time() - loading_start_time >= loading_duration:
            break

cv2.destroyWindow("Loading")  # Close the loading screen window
# Load highest scores and corresponding timestamps
try:
    with open(leaderboard_file, "r") as file:
        lines = file.readlines()
        highest_scores = [int(score.strip()) for score in lines[:2]]
        timestamps = [float(ts.strip()) for ts in lines[2:]]
except FileNotFoundError:
    highest_scores = [0, 0]
    timestamps = [0, 0]

# Generate a random initial direction for the ball
initial_direction = random.choice(["left", "right"])
if initial_direction == "left":
    ballPos = [40, 100]
    speedX = abs(speedX)  # Set the speed to positive
else:
    ballPos = [1200, 100]
    speedX = -abs(speedX)  # Set the speed to negative

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)
    timestamps = [0, 0]
    if timer_start is not None:
        time_remaining = max(0, duration - (time.time() - timer_start))
        timer_text = f"Time: {int(time_remaining)}s"
        text_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (img.shape[1] - text_size[0]) // 2
        cv2.putText(img, timer_text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Check for hands
        if hands:
            for hand in hands:
                x, y, w, h = hand['bbox']
                h1, w1, _ = imgBat1.shape
                y1 = y - h1 // 2
                y1 = np.clip(y1, 20, 415)

                if hand['type'] == "Left":
                    img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                    if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                        speedX = -speedX
                        ballPos[0] += 30
                        score[0] += 1
                        score_sound.play()

                if hand['type'] == "Right":
                    img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                    if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                        speedX = -speedX
                        ballPos[0] -= 30
                        score[1] += 1
                        score_sound.play()
        # Game Over
        if ballPos[0] < 40 or ballPos[0] > 1200:
            if game_over_time is None:
                game_over_time = time.time()  # Record the time when the game is over
                # Update highest scores and timestamps
                highest_scores[0] = max(highest_scores[0], score[0])
                highest_scores[1] = max(highest_scores[1], score[1])
                timestamps[0] = time.time()
                timestamps[1] = time.time()
                # Save highest scores and timestamps
                with open(leaderboard_file, "w") as file:
                    file.write(f"{highest_scores[0]}\n{highest_scores[1]}\n")
            gameOver = True

        if gameOver:
            if score[0] == score[1]:
                cv2.putText(img, "Equity!", (500, 360), cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)
            else:
                winner_text = f"Player {1 if score[0] > score[1] else 2} Wins!"
                loser_text = f"Player {2 if score[0] > score[1] else 1} Loses!"
                cv2.putText(img, winner_text, (400, 360), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 5)
                cv2.putText(img, loser_text, (400, 420), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 5)
            # Display leaderboard
            header_text = "Highest Scores"
            player1_text = f"Player 1: {highest_scores[0]}"
            player2_text = f"Player 2: {highest_scores[1]}"
            global_leaderboard = f"{header_text}\n{player1_text}\n{player2_text}"
            cv2.putText(img, global_leaderboard, (430, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.putText(img, "Press 'r' to restart or 'q' to quit", (430, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            timer_start = None
            # Check for 'R' key press
            key = cv2.waitKey(1)
            if key == ord('r') or key == ord('R'):
                ballPos = [100, 100]
                speedX = 15
                speedY = 15
                gameOver = False
                score = [0, 0]
                timer_start = time.time()  # Restart the timer
                game_over_time = None  # Reset the game over time
                imgGameOver = cv2.imread("Resources/gameOver.png")
        # If game not over move the ball
        else:
            # Move the Ball
            if ballPos[1] >= 500 or ballPos[1] <= 10:
                speedY = -speedY

            ballPos[0] += speedX
            ballPos[1] += speedY
            # Draw the ball
            img = cvzone.overlayPNG(img, imgBall, ballPos)
            cv2.putText(img, f"Player 1: {score[0]}", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.putText(img, f"Player 2: {score[1]}", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        resized_img = cv2.resize(imgRaw, (213, 120))[..., :3]
        rows, cols, _ = resized_img.shape
        roi = img[580:580 + rows, 20:20 + cols]
        if roi.shape[0] == rows and roi.shape[1] == cols:
            img[580:580 + rows, 20:20 + cols] = resized_img
        cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # Press 'q' or 'Esc' to exit
        break
