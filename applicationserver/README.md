For the networkserver the following steps were done to this point:

- Docker was installed. Here there was a slight problem with raspberryOS and the docker versioning, but after a quick googeling and the steps in this online instruction: ```https://www.wickedyoda.com/how-to-install-docker-on-raspberry-pi-os-trixie-2025-guide/``` docker was working.
- Chirpstackv4 was installed using Docker
- It can just be started via ```docker compose up -d``. This will launch everything, that is needed for the networkserver.


After start:

- Add the gateway (This needs a `name` and the `Gateway ID`). For me this resulted in a small conflict, because the `Gateway ID` from the `test_conf.json` file was already taken and that cause an error. The fix was to modify the `gateway_ID` and add it with the new one (Which I generated using chirpstack). 
- Then create an application.
- In the application you can create `devices`, but for that you first need to create a `Device Profile`. So you need to switch to the `Device Profiles` tab and add one. I added one with the base properties: ```{name: "profile01", Region: "EU868", MAC version: "LoRaWAN 1.1.0", "ADR Algorithm": "Default ADR algorithm (LoRa only)", "Expected uplink interval (secs)": 10}```. Later the tab `payload codec` will get interesting, but for now this config is enough.
- Now one can create a `Device`, in the application. Here the `name` is necessary and the `devEUI` (can be generated), the `joinEUI` (can be all zeros) and the `Application` and `Network` keys (can both be the same and generated) are important for the configuration of the nodes. The Application key should be kept secret.

After these initial configurations the connection of the nodes to the networkserver can be started. (The networkserver and the gateway need to be running)
Then the nodes, with the code, provided in the `node` folder can be started on the device.

Encoding and Decoding:

This is already enough to send uplinks and downlinks, but it is encrypted per default. This is where we come to the `payload codec` tab in the device profile. Here there a couple of options, but for this project the javascript function where the simplest. I used these two functions for `uplink_decryption` and `downlink_encryption`:

```
function decodeUplink(input) {
  var bytes = input.bytes;
  var i = 0;

  var counter = (bytes[i++] << 8) | bytes[i++];
  
  var tempRaw = (bytes[i++] << 8) | bytes[i++];
  if(tempRaw > 32767) tempRaw -= 65536;  // handle signed int16
  var temperature = tempRaw / 100.0;
  
  var status = bytes[i++];

  return {
    data: {
      counter:     counter,
      temperature: temperature,
      status:      status,
      sensor_ok:   (status & 0x01) !== 0
    }
  };
}
function encodeDownlink(input) {
  var data = input.data;
  var bytes = [];

  if(data.command === "set_interval") {
    bytes.push(0x01);
    bytes.push((data.interval >> 8) & 0xFF);
    bytes.push( data.interval       & 0xFF);
  }
  else if(data.command === "reset_counter") {
    bytes.push(0x02);
  }
  else if(data.command === "raw") {
    bytes = data.bytes;
  }

  return {
    bytes: bytes,
    fPort: 1
  };
}
```