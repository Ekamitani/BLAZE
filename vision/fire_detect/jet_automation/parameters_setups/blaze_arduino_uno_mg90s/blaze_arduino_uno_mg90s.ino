#include <Servo.h>

/*
  BLAZE - Controle de mira com dois servos MG90S
  Placa: Arduino Uno

  Serial Monitor:
    Baud: 115200
    Line ending: Newline

  Comandos:
    PING
    ID?
    STATUS?
    ZERO
    CENTER
    X90.0,Y85.5,P22.0

  Importante:
    O MG90S comum não mede a posição real.
    O status mostra a posição comandada pelo Arduino.

  Segurança de montagem:
    Na primeira utilização, deixe os atuadores/jato desacoplados.
    Ao ligar/reiniciar, este sketch posiciona X=90 e Y=90.
    Depois monte fisicamente o jato apontando para o centro da imagem.
*/

Servo servoX;
Servo servoY;

const int SERVO_X_PIN = 9;
const int SERVO_Y_PIN = 10;

const float SERVO_MIN = 0.0;
const float SERVO_MAX = 180.0;

const char* BLAZE_ID = "UNO_MG90S_V6";

float currentX = 90.0;
float currentY = 90.0;
float currentP = 22.0;

String buffer = "";

unsigned long lastStatusMillis = 0;
const unsigned long STATUS_INTERVAL_MS = 2000;

float clampFloat(float value, float minValue, float maxValue) {
  if (value < minValue) return minValue;
  if (value > maxValue) return maxValue;
  return value;
}

void applyServoAngles(float x, float y) {
  currentX = clampFloat(x, SERVO_MIN, SERVO_MAX);
  currentY = clampFloat(y, SERVO_MIN, SERVO_MAX);

  servoX.write((int)round(currentX));
  servoY.write((int)round(currentY));
}

void printStatusLine() {
  Serial.print("X=");
  Serial.print(currentX, 1);
  Serial.print(" ; Y=");
  Serial.print(currentY, 1);
  Serial.print(" ; P=");
  Serial.print(currentP, 1);
  Serial.print(" ; ID=");
  Serial.println(BLAZE_ID);
}

void printOk() {
  Serial.print("OK ");
  printStatusLine();
}

bool parseCommand(String cmd) {
  cmd.trim();

  if (cmd.length() == 0) {
    return true;
  }

  if (cmd == "PING") {
    Serial.print("BLAZE_OK ");
    Serial.println(BLAZE_ID);
    return true;
  }

  if (cmd == "ID?") {
    Serial.print("BLAZE_ID ");
    Serial.println(BLAZE_ID);
    return true;
  }

  if (cmd == "STATUS?") {
    printStatusLine();
    return true;
  }

  if (cmd == "ZERO") {
    applyServoAngles(0.0, 0.0);
    printOk();
    return true;
  }

  if (cmd == "CENTER") {
    applyServoAngles(90.0, 90.0);
    printOk();
    return true;
  }

  int xIndex = cmd.indexOf('X');
  int yIndex = cmd.indexOf('Y');
  int pIndex = cmd.indexOf('P');

  int comma1 = cmd.indexOf(',');
  int comma2 = cmd.indexOf(',', comma1 + 1);

  if (xIndex < 0 || yIndex < 0 || comma1 < 0) {
    return false;
  }

  String xStr = cmd.substring(xIndex + 1, comma1);
  String yStr = "";

  if (comma2 > comma1) {
    yStr = cmd.substring(yIndex + 1, comma2);
  } else {
    yStr = cmd.substring(yIndex + 1);
  }

  if (pIndex >= 0) {
    String pStr = cmd.substring(pIndex + 1);
    currentP = pStr.toFloat();
  }

  float x = xStr.toFloat();
  float y = yStr.toFloat();

  applyServoAngles(x, y);
  printOk();

  return true;
}

void setup() {
  Serial.begin(115200);

  servoX.attach(SERVO_X_PIN);
  servoY.attach(SERVO_Y_PIN);

  applyServoAngles(90.0, 90.0);

  delay(300);

  Serial.println("BLAZE Arduino pronto.");
  Serial.print("ID=");
  Serial.println(BLAZE_ID);
  Serial.println("Baud=115200");
  Serial.println("Formato status: X=valor ; Y=valor ; P=valor");
  Serial.println("Comandos: PING, ID?, STATUS?, ZERO, CENTER, X90.0,Y85.5,P22.0");

  printStatusLine();
  lastStatusMillis = millis();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (buffer.length() > 0) {
        bool ok = parseCommand(buffer);

        if (!ok) {
          Serial.print("ERRO comando: ");
          Serial.println(buffer);
        }

        buffer = "";
      }
    } else {
      buffer += c;

      if (buffer.length() > 80) {
        buffer = "";
        Serial.println("ERRO buffer serial muito longo.");
      }
    }
  }

  if (millis() - lastStatusMillis >= STATUS_INTERVAL_MS) {
    printStatusLine();
    lastStatusMillis = millis();
  }
}
