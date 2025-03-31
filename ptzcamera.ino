// Import libraries 
#include <ESP8266WiFi.h>
#include <ESPAsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <Servo.h>


const char *ssid = "REPLACE_WITH_YOUR_SSID";         // replace with your SSID
const char *password = "REPLACE_WITH_YOUR_PASSWORD"; // replace with your Password
const uint8_t servoPin0 = D0;                         // replace with servo pin
const uint8_t servoPin1 = D1;                         // replace with servo pin

/* Create Servo Object */
Servo servo0;
Servo servo1;
// Create Server instance
AsyncWebServer server(80);


const char* html = R"rawliteral(

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP8266 Servo Control</title>
    <link rel="shortcut icon" href="favicon.ico" type="image/x-icon">
    <style>
        * {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }

        body, html {
            height: 100%;
            display: block;
            background-color: #f0f0f0;
        }

        .container {
            text-align: center;
            border: 1px solid #ccc;
            padding: 20px;
            background-color: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            width: 400px;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .container h1 {
            margin-bottom: 20px;
        }

        .custom-range {
            width: 100%;
        }

        .display-4 {
            font-size: 1.5rem;
        }

        div {
            text-decoration: underline;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1 class="display-4">ESP8266 Servo Control</h1><br>
        
        <!-- Angle 1 Slider -->
        <h3>Angle 1</h3>
        <input type="range" id="range1" min="0" max="180" value="0" class="custom-range">
        <p id="currentAngle1" class="display-4">Current Value: 0 °</p><br><br>

        <!-- Angle 2 Slider -->
        <h3>Angle 2</h3>
        <input type="range" id="range2" min="0" max="180" value="0" class="custom-range">
        <p id="currentAngle2" class="display-4">Current Value: 0 °</p><br><br>
    </div>

    <script>
        // Function to update the display and send POST request
        function updateAngleDisplayAndSend(sliderId, displayId, url) {
            const angleValue = document.getElementById(sliderId).value;
            document.getElementById(displayId).textContent = "Current Value: " + angleValue + "°";

            // Send the angle to the server
            const xhr = new XMLHttpRequest();
            xhr.open("POST", url, true);
            xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
            xhr.send("angle=" + angleValue);
        }

        // Add event listeners for both sliders
        document.getElementById("range1").addEventListener("input", function () {
            updateAngleDisplayAndSend("range1", "currentAngle1", "/angle0");
        });

        document.getElementById("range2").addEventListener("input", function () {
            updateAngleDisplayAndSend("range2", "currentAngle2", "/angle1");
        });
    </script>
</body>

</html>
)rawliteral";


void setup()
{
  // Attach Servo, start SPIFFS and Connect to WiFi
  Serial.begin(115200);
  servo0.attach(servoPin0);
  servo1.attach(servoPin1);
  if (!SPIFFS.begin())
  {
    Serial.println("An Error has occurred while mounting SPIFFS");
    return;
  }
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi..");
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(1000);
    Serial.print(".");
  }
  Serial.print("\nConnected to the WiFi network: ");
  Serial.print(WiFi.SSID());
  Serial.print("IP address:");
  Serial.print(WiFi.localIP());

  String str = "";
Dir dir = SPIFFS.openDir("/");
while (dir.next()) {
    str += dir.fileName();
    str += " / ";
    str += dir.fileSize();
    str += "\r\n";
}
Serial.print(str);

  // Send home page from SPIFFS
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    //Serial.println("Send index.html.");
    //request->send(SPIFFS, "/index.html", "text/html");
    request->send(200, "text/html", html);
  });
  // Receive Angle from client and process it 
  server.on("/angle0", HTTP_POST, [](AsyncWebServerRequest *request) {
    String angle0 = request->arg("angle");
    Serial.println("Current Position: " + angle0 + "°");
    servo0.write(angle0.toInt());
    request->send(200);
  });

  server.on("/angle1", HTTP_POST, [](AsyncWebServerRequest *request) {
    String angle1 = request->arg("angle");
    Serial.println("Current Position: " + angle1 + "°");
    //servo1.writeMicroseconds(angle1.toInt()*10);
    servo1.write(angle1.toInt());
    request->send(200);
  });
  // Send Favicon 
  server.serveStatic("/favicon.ico", SPIFFS, "/favicon.ico");
  // Begin Server
  server.begin();
}
void loop()
{
}
