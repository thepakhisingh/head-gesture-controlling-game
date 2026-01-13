import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import time
import math
import sys

# CAMERA
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
    sys.exit()

#  MEDIAPIPE 
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

NOSE_ID = 1
LEFT_EAR = [159, 145, 33, 133]
RIGHT_EAR = [386, 374, 362, 263]

# PYGAME 
pygame.init()
WIDTH, HEIGHT = 600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Head Shooter 💗")
clock = pygame.time.Clock()

FONT_BIG = pygame.font.SysFont(None, 56)
FONT_MED = pygame.font.SysFont(None, 42)

# PLAYER 
player_x = WIDTH // 2
player_y = HEIGHT - 80
player_w, player_h = 60, 40
target_x = player_x

DEAD_ZONE = 25
MAX_SPEED = 10
SMOOTHING = 0.08

# BULLETS / ENEMIES
bullets = []
enemies = []

def new_enemy():
    return pygame.Rect(random.randint(20, WIDTH-60), -40, 50, 40)

#  CALIBRATION
calibrated = False
calib_vals = []
calib_start = time.time()
neutral_head_x = 0

# BLINK 
def eye_ratio(eye):
    A = math.dist(eye[0], eye[1])
    B = math.dist(eye[2], eye[3])
    return A / (B + 1e-6)

blink_cd = 0

# GAME STATE
running = True
game_over = False
score = 0

# HEARTS
lives = 3
INVINCIBLE_TIME = 60
invincible_timer = 0

#  HUD FUNCTION 
def draw_center_hud(surface, score, lives):
    hud = pygame.Surface((WIDTH, 150), pygame.SRCALPHA)
    hud.set_alpha(100)  # opacity (0–255)

    pink = (255, 105, 180)  # hot pink

    score_text = FONT_BIG.render(f"SCORE {score}", True, pink)
    heart_text = FONT_MED.render("❤️ " * lives, True, pink)

    hud.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 10))
    hud.blit(heart_text, (WIDTH//2 - heart_text.get_width()//2, 70))

    surface.blit(hud, (0, HEIGHT//2 - 75))

# GAME LOOP
while running:
    clock.tick(60)
    screen.fill((18, 18, 18))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q:
                running = False
            if e.key == pygame.K_r and game_over:
                bullets.clear()
                enemies.clear()
                score = 0
                lives = 3
                game_over = False

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks and not game_over:
        face = res.multi_face_landmarks[0].landmark

        # HEAD MOVEMENT
        nose_x = face[NOSE_ID].x * w

        if not calibrated:
            calib_vals.append(nose_x)
            if time.time() - calib_start > 3:
                neutral_head_x = np.mean(calib_vals)
                calibrated = True
        else:
            diff = nose_x - neutral_head_x
            if abs(diff) > DEAD_ZONE:
                target_x += np.clip(diff * 0.12, -MAX_SPEED, MAX_SPEED)

        target_x = max(0, min(WIDTH - player_w, target_x))

        # BLINK SHOOT
        l_ear = eye_ratio([(face[i].x*w, face[i].y*h) for i in LEFT_EAR])
        r_ear = eye_ratio([(face[i].x*w, face[i].y*h) for i in RIGHT_EAR])
        if (l_ear + r_ear)/2 < 0.20 and blink_cd == 0:
            bullets.append(
                pygame.Rect(player_x + player_w//2 - 5, player_y - 10, 10, 10)
            )
            blink_cd = 15

    if blink_cd > 0:
        blink_cd -= 1

    # PLAYER MOVE
    player_x = player_x * (1 - SMOOTHING) + target_x * SMOOTHING
    player = pygame.Rect(player_x, player_y, player_w, player_h)

    if invincible_timer > 0:
        invincible_timer -= 1

    if not game_over:
        for b in bullets[:]:
            b.y -= 10
            if b.y < 0:
                bullets.remove(b)

        if random.randint(1, 40) == 1:
            enemies.append(new_enemy())

        for en in enemies[:]:
            en.y += 5

            if (en.colliderect(player) or en.y > HEIGHT) and invincible_timer == 0:
                lives -= 1
                invincible_timer = INVINCIBLE_TIME
                enemies.remove(en)
                if lives <= 0:
                    game_over = True
                continue

            for b in bullets[:]:
                if en.colliderect(b):
                    enemies.remove(en)
                    bullets.remove(b)
                    score += 1
                    break

    # DRAW GAME
    if invincible_timer % 10 < 5:
        pygame.draw.rect(screen, (0, 140, 255), player)

    for b in bullets:
        pygame.draw.circle(screen, (255, 255, 0), b.center, 6)
    for en in enemies:
        pygame.draw.rect(screen, (255, 0, 0), en)

    # heart CENTER HUD
    draw_center_hud(screen, score, lives)

    if not calibrated:
        t = FONT_MED.render("CALIBRATING – KEEP HEAD STILL", True, (255, 180, 200))
        screen.blit(t, (WIDTH//2 - t.get_width()//2, HEIGHT//2 - 140))

    if game_over:
        t1 = FONT_BIG.render("GAME OVER", True, (255, 90, 120))
        t2 = FONT_MED.render("R = Restart   Q = Quit", True, (255, 200, 220))
        screen.blit(t1, (WIDTH//2 - t1.get_width()//2, HEIGHT//2 + 100))
        screen.blit(t2, (WIDTH//2 - t2.get_width()//2, HEIGHT//2 + 150))

    pygame.display.update()
#try it out and if you find any difficulty please contanct me or see codewithpakhi.com hehehe
cap.release()
pygame.quit()
cv2.destroyAllWindows()

