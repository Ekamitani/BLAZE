
#include <Servo.h>

/*
  BLAZE - Controle de mira com dois servos MG90S
  Protocolo serial compatível com o notebook de referência:
    X90.0,Y85.5,P22.0,W0
*/

Servo servoX;
Servo servoY;

const int SERVO_X_PIN = 9;
const int SERVO_Y_PIN = 10;
const int WATER_PIN = 8;
const float SERVO_MIN = 0.0;
const float SERVO_MAX = 180.0;
const char* BLAZE_ID = "UNO_MG90S_RGB_MONO_V8";

float currentX = 90.0;
float currentY = 90.0;
float currentP = 22.0;
bool waterOn = false;
String buffer = "";
unsigned long lastStatusMillis = 0;
const unsigned long STATUS_INTERVAL_MS = 2000;
const unsigned long WATER_TIMEOUT_MS = 1000;
unsigned long lastCommandMillis = 0;

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

void applyWater(bool state) {
  waterOn = state;
  digitalWrite(WATER_PIN, waterOn ? HIGH : LOW);
}

void printStatusLine() {
  Serial.print("X="); Serial.print(currentX, 1);
  Serial.print(" ; Y="); Serial.print(currentY, 1);
  Serial.print(" ; P="); Serial.print(currentP, 1);
  Serial.print(" ; W="); Serial.print(waterOn ? 1 : 0);
  Serial.print(" ; ID="); Serial.println(BLAZE_ID);
}

void printOk() {
  Serial.print("OK ");
  printStatusLine();
}

String valueUntilCommaOrEnd(String cmd, int startIndex) {
  int comma = cmd.indexOf(',', startIndex);
  if (comma < 0) return cmd.substring(startIndex);
  return cmd.substring(startIndex, comma);
}

bool parseCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return true;

  if (cmd == "PING") { Serial.print("BLAZE_OK "); Serial.println(BLAZE_ID); return true; }
  if (cmd == "ID?") { Serial.print("BLAZE_ID "); Serial.println(BLAZE_ID); return true; }
  if (cmd == "STATUS?") { printStatusLine(); return true; }
  if (cmd == "ZERO") { applyServoAngles(0.0, 0.0); applyWater(false); printOk(); return true; }
  if (cmd == "CENTER") { applyServoAngles(90.0, 90.0); applyWater(false); printOk(); return true; }

  int wIndex = cmd.indexOf('W');
  if (wIndex >= 0) {
    String wStr = valueUntilCommaOrEnd(cmd, wIndex + 1);
    applyWater(wStr.toInt() == 1);
  }

  int xIndex = cmd.indexOf('X');
  int yIndex = cmd.indexOf('Y');
  int pIndex = cmd.indexOf('P');

  if (xIndex >= 0 && yIndex >= 0) {
    String xStr = valueUntilCommaOrEnd(cmd, xIndex + 1);
    String yStr = valueUntilCommaOrEnd(cmd, yIndex + 1);
    applyServoAngles(xStr.toFloat(), yStr.toFloat());
  }

  if (pIndex >= 0) {
    String pStr = valueUntilCommaOrEnd(cmd, pIndex + 1);
    currentP = pStr.toFloat();
  }

  if (xIndex >= 0 || yIndex >= 0 || pIndex >= 0 || wIndex >= 0) {
    lastCommandMillis = millis();
    printOk();
    return true;
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  servoX.attach(SERVO_X_PIN);
  servoY.attach(SERVO_Y_PIN);
  pinMode(WATER_PIN, OUTPUT);
  applyWater(false);
  applyServoAngles(90.0, 90.0);
  delay(300);
  Serial.println("BLAZE Arduino pronto.");
  Serial.print("ID="); Serial.println(BLAZE_ID);
  Serial.println("Baud=115200");
  Serial.println("Comandos: PING, ID?, STATUS?, ZERO, CENTER, X90.0,Y85.5,P22.0,W0");
  printStatusLine();
  lastStatusMillis = millis();
  lastCommandMillis = millis();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (buffer.length() > 0) {
        bool ok = parseCommand(buffer);
        if (!ok) { Serial.print("ERRO comando: "); Serial.println(buffer); }
        buffer = "";
      }
    } else {
      buffer += c;
      if (buffer.length() > 100) { buffer = ""; Serial.println("ERRO buffer serial muito longo."); }
    }
  }

  if (waterOn && millis() - lastCommandMillis > WATER_TIMEOUT_MS) {
    applyWater(false);
  }

  if (millis() - lastStatusMillis >= STATUS_INTERVAL_MS) {
    printStatusLine();
    lastStatusMillis = millis();
  }
}
