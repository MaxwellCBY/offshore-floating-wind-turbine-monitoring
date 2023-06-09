import streamlit as st
from pynmeagps.nmeareader import NMEAReader
import pynmeagps.exceptions as nme
import sys
import random
import simplekml
import asyncio
import base64
import io
import numpy as np
import pandas as pd
import pydeck as pdk
from PIL import Image
import os

image1 = Image.open(os.path.join(os.path.dirname(__file__), 'GPS-Log-Messung1-Course.PNG'))
image2 = Image.open(os.path.join(os.path.dirname(__file__), 'GPS.PNG'))

class SatData:
    def __init__(self, id, elevation, azimuth, snr):
        self.id = id
        self.elevation = elevation
        self.azimuth = azimuth
        self.snr = snr
        self.pdop = 0
        self.hdop = 0
        self.vdop = 0
        self.navmode = 0
        #print("Created sat with id: {} elevation: {} azimuth: {} snr: {}".format(self.id,self.elevation,self.azimuth,self.snr))
    def setPDOP(self, val):
        self.pdop = val
    
    def setHDOP(self, val):
        self.hdop = val

    def setVDOP(self, val):
        self.vdop = val
    
    def setNavMode(self, val):
        self.navmode = val

    def getPDOP(self):
        return self.pdop

    def getHDOP(self):
        return self.hdop

    def getVDOP(self):
        return self.vdop

    def getNavMode(self):
        return self.navmode

    def getId(self):
        return self.id

    def getElevation(self):
        return self.elevation
    
    def getAzimuth(self):
        return self.azimuth
    
    def getSNR(self):
        return self.snr

rssiDict = {}
rssiList = []
line = 0
minLat = sys.maxsize
maxLat = -sys.maxsize
minLon = sys.maxsize
maxLon = -sys.maxsize
time = 0
hdop = 0
vdop = 0
pdop = 0
navmode = 0
kmlFileName = "output.kml"

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = io.BytesIO()
	img.save(buffered, format="PNG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href

def text_downloader(raw_text, fileName):
	b64 = base64.b64encode(raw_text.encode()).decode()

	st.markdown("#### Download File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{fileName}">{fileName}</a>'
	st.markdown(href,unsafe_allow_html=True)

#http://kml4earth.appspot.com/icons.html
def saveKML(_rssiDict, fileName):
    kml = simplekml.Kml()
    
    for key, values in _rssiDict.items():
        snr = 0
        snrAverage = 0
        satDescription = "<strong> {} </strong></br>".format(key)
        satCount = len(_rssiDict[key][2])
        gpsMode = 0
        #list of satData
        if(satCount > 0):
            first = 0
            for satData in _rssiDict[key][2]:
                snr += satData.getSNR()
                if(not first):
                    satDescription += "<li>SatCount: {} </li> <li>NavMode: {}</li><li>PDOP: {}</li> <li>HDOP: {}</li><li>VDOP: {}</li></ul>".format(satCount,satData.getNavMode(),satData.getPDOP(), satData.getHDOP(), satData.getVDOP())
                    gpsMode = satData.getNavMode()
                    first = 1

                satDescription += f"<ul><li><strong>Satellite ID: {satData.getId()}</strong></li> <li>Elevation: {satData.getElevation()}</li> <li>Azimuth {satData.getAzimuth()}</li> <li>SNR: {satData.getSNR()}</li>  </ul>"
            
            snrAverage = snr / satCount
        
        snrAverage = round(snrAverage)
        satPnt = kml.newpoint(name=str(snrAverage), coords=[(_rssiDict[key][0],_rssiDict[key][1])])
        satPnt.description = "{}".format(satDescription)

        if gpsMode != 3 :
            satPnt.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/caution.png"
        elif(snrAverage >= 30):
            satPnt.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/blu-square-lv.png'
        elif(snrAverage > 20 and snrAverage <= 29):
            satPnt.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/grn-square-lv.png'
        else:
            satPnt.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-square-lv.png'
        
    text_downloader(kml.kml(),fileName)
    #kml.save("./temp.kml")

def showMap(_rssiDict):
    df = pd.DataFrame.from_dict(_rssiDict, columns=['lon', 'lat', 'sat'], orient='index')

    view = pdk.data_utils.compute_view(df[['lon', 'lat']])
    view.zoom = 12
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view,
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["lon", "lat"],
                auto_highlight=True,
                get_radius=10,          # Radius is given in meters
                get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
            ),
        ]
    ))

st.title("GPS NMEA to KML")
st.write("Provide a NMEA GPS log for processing")

nmeafile = st.file_uploader("Upload NMEA GPS log", type=['log', 'nmea'])

submit = st.button("Submit")

def addPos(lat, lon, time, _rssiList):
        rssiDict[time] = (lon,lat,_rssiList.copy())

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

if submit:
    nmeaError = False
    if nmeafile is not None:
        st.write("File submitted")
        file_details = {"FileName":nmeafile.name,"FileType":nmeafile.type,"FileSize":nmeafile.size}
        st.write(file_details)
        nmr = NMEAReader(nmeafile, nmeaonly=False)

        try:
            for (raw_data, msg) in nmr: 
                if msg:
                    
                    #use to debug bad nmea line
                    #print(line)
                    line += 1
                    #print(msg)
                    if(msg.msgID == "GSA"):
                        if( hasattr(msg, 'PDOP')):
                            pdop = msg.PDOP
                        if( hasattr(msg, 'HDOP')):
                            hdop = msg.HDOP
                        if( hasattr(msg, 'VDOP')):
                            vdop = msg.VDOP
                        if( hasattr(msg,'navMode')):
                            navmode = msg.navMode

                    if msg.msgID == "GSV":
                        #handle sattelite count
                        if(hasattr(msg, 'svid_01') and msg.svid_01 and msg.cno_01):
                            satData = SatData(msg.svid_01,msg.elv_01, msg.az_01,msg.cno_01)
                            satData.setHDOP(hdop)
                            satData.setVDOP(vdop)
                            satData.setPDOP(pdop)
                            satData.setNavMode(navmode)
                            rssiList.append(satData)
                        if(hasattr(msg, 'svid_02') and msg.svid_02 and msg.cno_02):
                            satData = SatData(msg.svid_02,msg.elv_02, msg.az_02,msg.cno_02)
                            satData.setHDOP(hdop)
                            satData.setVDOP(vdop)
                            satData.setPDOP(pdop)
                            satData.setNavMode(navmode)
                            rssiList.append(satData)
                        if( hasattr(msg, 'svid_03') and msg.svid_03 and msg.cno_03):
                            satData = SatData(msg.svid_03,msg.elv_03, msg.az_01,msg.cno_03)
                            satData.setHDOP(hdop)
                            satData.setVDOP(vdop)
                            satData.setPDOP(pdop)
                            satData.setNavMode(navmode)
                            rssiList.append(satData)
                        if(hasattr(msg, 'svid_04') and msg.svid_04 and msg.cno_04):
                            satData = SatData(msg.svid_04,msg.elv_04, msg.az_04,msg.cno_04)
                            satData.setHDOP(hdop)
                            satData.setVDOP(vdop)
                            satData.setPDOP(pdop)
                            satData.setNavMode(navmode)
                            rssiList.append(satData)
        
                    elif msg.msgID == "GGA":
                        if( hasattr(msg, 'time') and msg.time):
                            time = msg.time
                        


                        if(hasattr(msg, 'lat') and msg.lat):
                            if(hasattr(msg, 'lon') and msg.lon):
                                if(msg.lat > maxLat):
                                    maxLat = msg.lat
                                if(msg.lat < minLat):
                                    minLat = msg.lat

                                if(msg.lon > maxLon):
                                    maxLon = msg.lon
                                if(msg.lon < minLon):
                                    minLon = msg.lon
                                
                        
                                addPos(msg.lat,msg.lon, time, rssiList)
                        
                        rssiList.clear()
                else:
                    st.write(f"Warning: Unable to parse some lines.  Please check file for errors")
                    # use default file for demo
                    


        except (nme.NMEAStreamError, nme.NMEAMessageError, nme.NMEATypeError, nme.NMEAParseError) as err:
                st.write(f"Error: Unable to parse NMEA {err}")
                nmeaError = True

        if not nmeaError:             
            fnameSplit = str.split(nmeafile.name,'.')
            if len(fnameSplit) >=1:
                kmlFileName = f"{fnameSplit[0]}.kml"
                
            bbox = maxLon,maxLat,minLon,minLat
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            saveKML(rssiDict, kmlFileName)
            showMap(rssiDict)
        
        st.image(image1)
    else:
        st.image(image2)
        st.write("No file was uploaded!")




