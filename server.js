const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const { spawn } = require("child_process");

const app = express();
const server = http.createServer(app);
const io = new Server(server);
const port = 3000;

// Serve static files
app.get("/", (req, res) => {
  res.sendFile(__dirname + "/index.html");
});

// Spawn Python process with unbuffered output (-u)
const pythonProcess = spawn("python", ["-u", "ai_processor.py"]);

// Handle data stream from Python
let dataBuffer = "";
pythonProcess.stdout.on("data", (data) => {
  dataBuffer += data.toString();

  // Process complete lines
  let lines = dataBuffer.split("\n");
  dataBuffer = lines.pop(); // Keep incomplete chunk

  for (let line of lines) {
    line = line.trim();

    // [Core] Filter valid JSON packets prefixed with "DATA_START:"
    if (line.startsWith("DATA_START:")) {
      try {
        const jsonStr = line.substring(11); // Remove prefix
        const jsonData = JSON.parse(jsonStr);

        io.emit("video_frame", jsonData);
      } catch (e) {
        console.error("JSON Parsing Error:", e);
      }
    } else if (line.length > 0) {
      // Log debug messages from Python
      console.log(`[Python Log]: ${line}`);
    }
  }
});

// Handle stderr (errors & logs)
pythonProcess.stderr.on("data", (data) => {
  console.error(`Python Error: ${data}`);
});

server.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
