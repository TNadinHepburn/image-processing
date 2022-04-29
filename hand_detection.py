import cv2
import mediapipe as mp

class landmarkCoordinates(object):
    def __init__(self,x,y,z):
        self.X = x
        self.Y = y
        self.Z = z

    def __add__(self,other):
        return [self.X + other.getX(), self.Y + other.getY(), self.Z + other.getZ()]

    def __mul__(self,other):
        return self.X * other.X+ self.Y * other.Y+ self.Z * other.Z

    def __str__(self):
        return(str(self.X)+", "+str(self.Y)+", "+str(self.Z))
        
    def getXYZ(self):
        return [self.X,self.Y,self.Z]

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def getZ(self):
        return self.Z

class landmark_vector(object):
    def __init__(self,magnitude,direction):
        self.Magnitude = magnitude
        self.Direction = direction

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
EXTRACTED_DATA = []
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

for label in labels:
    IMAGE_FILES = []
    for i in range(1,3001):
        IMAGE_FILES.append(f"{label}\\{label}{str(i)}.jpg")

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        for idx, file in enumerate(IMAGE_FILES):
            landmarkCoordinates = []
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                continue
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarkCoordinates.append([landmark.x,landmark.y,landmark.z])
            EXTRACTED_DATA.append(landmarkCoordinates)

        with open(f'landmark_data\\{label}.txt', 'a') as data_file:
            data_file.write(str(EXTRACTED_DATA)+"\n")










        # wristToThumbBase = [one_image[0],one_image[1]]
        # wristToIndexBase = [one_image[0],one_image[5]]
        # wristToLittleBase = [one_image[0],one_image[17]]
        # thumbBaseToThumbTip = [one_image[1],one_image[4]]
        # indexBaseToIndexTip = [one_image[5],one_image[8]]
        # middleBaseToMiddleTip = [one_image[9],one_image[12]]
        # ringBaseToRingTip = [one_image[13],one_image[16]]
        # littleBaseToLittleTip = [one_image[17],one_image[20]]









        
        # for hand_landmarks in results.multi_hand_landmarks:
        #     for landmark in hand_landmarks.landmark:
        #         class_one_image.append(landmark_coordinates(landmark.x,landmark.y,landmark.z))


        # ###
        # wristToThumbBase = [class_one_image[1]+class_one_image[0]]
        # wristToIndexBase = [class_one_image[5]+class_one_image[0]]
        # wristToLittleBase = [class_one_image[17]+class_one_image[0]]
        # ###

        # ###
        # thumbBaseToThumbPIP = [class_one_image[2]+class_one_image[1]]
        # thumbPIPToThumbDIP = [class_one_image[3]+class_one_image[2]]
        # thumbDIPToThumbTip = [class_one_image[4]+class_one_image[3]]
        # ###

        # ###
        # indexBaseToIndexPIP = [class_one_image[6]+class_one_image[5]]
        # indexPIPToIndexDIP = [class_one_image[7]+class_one_image[6]]
        # indexDIPToIndexTip = [class_one_image[8]+class_one_image[7]]
        # ###

        # ###
        # middleBaseToMiddlePIP = [class_one_image[10]+class_one_image[9]]
        # middlePIPToMiddleDIP = [class_one_image[11]+class_one_image[10]]
        # middleDIPToMiddleTip = [class_one_image[12]+class_one_image[11]]
        # ###

        # ###
        # ringBaseToRingPIP = [class_one_image[14]+class_one_image[13]]
        # ringPIPToRingDIP = [class_one_image[15]+class_one_image[14]]
        # ringDIPToRingTip = [class_one_image[16]+class_one_image[15]]
        # ###

        # ###
        # littleBaseToLittlePIP = [class_one_image[18]+class_one_image[17]]
        # littlePIPToLittleDIP = [class_one_image[19]+class_one_image[18]]
        # littleDIPToLittleTip = [class_one_image[20]+class_one_image[19]]
        # ###

        # fingercoords = []
        # for i in range (5):
        #     changesx = one_image[0][0]+one_image[i*4+1][0]+one_image[i*4+2][0]+one_image[i*4+3][0]+one_image[i*4+4][0]
        #     changesy = one_image[0][1]+one_image[i*4+1][1]+one_image[i*4+2][1]+one_image[i*4+3][1]+one_image[i*4+4][1]
        #     changesz = one_image[0][2]+one_image[i*4+1][2]+one_image[i*4+2][2]+one_image[i*4+3][2]+one_image[i*4+4][2]
        #     fingercoords.append([changesx,changesy,changesz])
        # print(fingercoords)




# import cv2
# import mediapipe as mp
# cap = cv2.VideoCapture(0)
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
# while True:
#     sucess, image = cap.read()
#     imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     results = hands.process(imageRGB)
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 h,w,c = image.shape
#                 cx,cy = int(lm.x * w), int(lm.y * h)
#             mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
#             cv2.imshow("Output", image)
#             print(results.multi_hand_landmarks)
#             cv2.waitKey(1)

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# # For static images:
# IMAGE_FILES = ["archive\\asl_alphabet_train\\asl_alphabet_train\\B\\B3.jpg","archive\\asl_alphabet_train\\asl_alphabet_train\\B\\B659.jpg"]
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=1,
#     min_detection_confidence=0.5) as hands:

#     for idx, file in enumerate(IMAGE_FILES):
#         one_image = []
#         # Read an image, flip it around y-axis for correct handedness output (see
#         # above).
#         image = cv2.flip(cv2.imread(file), 1)
#         # Convert the BGR image to RGB before processing.
#         results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         # Print handedness and draw hand landmarks on the image.
#         print('Handedness:', results.multi_handedness)
#         if not results.multi_hand_landmarks:
#             continue
#         for hand_landmarks in results.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 one_image.append([landmark.x,landmark.y,landmark.z])
#         print(one_image)
#         image_height, image_width, _ = image.shape
#         annotated_image = image.copy()
#         for hand_landmarks in results.multi_hand_landmarks:
#             print('hand_landmarks:', hand_landmarks)
#             print(
#                 f'Index finger tip coordinates: (',
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#             )
#             mp_drawing.draw_landmarks(
#                 annotated_image,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())


    
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#         continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#         mp_drawing.plot_landmarks(
#             hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)