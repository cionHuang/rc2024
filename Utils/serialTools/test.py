import usbToTTL

port = usbToTTL.TTL()

try:
    
    while True:
        print(port.portReading())
        
finally:
    port.killport()

