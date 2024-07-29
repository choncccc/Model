const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', ws => {
   //console.log('Client connected');

  ws.on('message', message => {
    console.log(`Received: ${message}`);
    wss.clients.forEach(client => {
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(message.toString());
      }
    });
  });

  ws.on('close', () => {
    //console.log('Client disconnected');
  });
});