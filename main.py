import cv2  # Biblioteca para acessar a câmera e mostrar o vídeo
import mediapipe as mp  # Biblioteca para detectar os pontos do corpo (landmarks)
import time  # Para medir o tempo

# Inicializa o detector de pose do Mediapipe
mp_pose = mp.solutions.pose  # Atalho para a solução de pose
mp_drawing = mp.solutions.drawing_utils  # Atalho para desenhar os landmarks

# Função principal que acessa a câmera e processa a detecção de postura
def detect_posture():
    cap = cv2.VideoCapture(0)  # Acessa a webcam (0 significa a primeira câmera)
    start_time = time.time()  # Tempo inicial para controle de erros

    with mp_pose.Pose() as pose:
        while cap.isOpened():  # Mantém o loop enquanto a câmera estiver aberta
            ret, frame = cap.read()  # Captura cada frame do vídeo
            if not ret:
                break  # Sai do loop se não conseguir capturar o vídeo

            # Converte o frame de BGR para RGB (necessário para o Mediapipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Faz a detecção de landmarks
            result = pose.process(rgb_frame)

            if result.pose_landmarks:  # Se pontos do corpo foram detectados
                # Desenha os pontos do corpo na imagem original
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Obtém as coordenadas dos ombros e cabeça
                left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                head = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                # Lógica para checar postura dos ombros
                shoulders_level = abs(left_shoulder.y - right_shoulder.y) < 0.05  # Verifica se os ombros estão nivelados
                head_straight = head.y < left_shoulder.y and head.y < right_shoulder.y  # Cabeça deve estar acima dos ombros

                # Mensagens de postura
                if not shoulders_level:
                    cv2.putText(frame, 'Erro: Ombros em abducao!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Vermelho
                    start_time = time.time()  # Reinicia o contador se houver erro
                elif not head_straight:
                    cv2.putText(frame, 'Erro: Cabeca inclinada!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Vermelho
                    start_time = time.time()
                else:
                    elapsed_time = time.time() - start_time  # Tempo desde a última postura correta
                    if elapsed_time > 5:  # Se estiver 5 segundos em postura correta
                        cv2.putText(frame, 'Postura Correta!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Verde

            # Exibe o vídeo com os pontos desenhados
            cv2.imshow('Deteccao de Postura', frame)

            # Se apertar 'q', sai do loop e fecha a câmera
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()  # Fecha a captura da câmera
    cv2.destroyAllWindows()  # Fecha todas as janelas

# Executa a função
detect_posture()
