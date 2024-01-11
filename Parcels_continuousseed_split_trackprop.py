#!/usr/bin/env python
# coding: utf-8

# Lagrangian tracking with the Glorys dataset
# Continuously seed new particles
# Track water properties
# Include initial volume transport
# Include water velocity

import numpy as np
from pyproj import Geod
from netCDF4 import  MFDataset, Dataset
from parcels import Field,FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4_3D, AdvectionDiffusionM1
import pprint
from datetime import timedelta
from operator import attrgetter
import xarray as xr
from random import uniform
from os import listdir
import cartopy.crs as ccrs
from os import listdir
from parcels import ParticleFile
from datetime import timedelta as delta
from math import cos, atan, atan2, radians

# Kernel to track the age of particles
def ageing(particle, fieldset, time):
    particle.age += 1./144.

# Kernel to kill particles older than 3 years
def killold(particle, fieldset, time):
    if particle.age > 365.*3. : #delete after X days
        particle.delete()
    if particle.stuck > 1440*10: # 10 days in # of time steps
        particle.delete()

class ocean_particle(JITParticle):
    #add some variables
    age = Variable('age',dtype=np.float32, initial=0.)
    stuck = Variable('stuck', dtype=np.int32, initial=0.)
    prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon'))  # the previous longitude
    prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat'))  # the previous latitude.
    temperature = Variable('temperature', dtype=np.float32)
    salinity = Variable('salinity', dtype=np.float32)
    uvel = Variable('u', dtype=np.float32)
    vvel = Variable('v', dtype=np.float32)
    wvel = Variable('w', dtype=np.float32)

    area = Variable('area', dtype=np.float32, to_write='once', initial=0.)
    # It would be much better if we could store it only at the first time step, but at the first time step u=0 so volume=0...
    volume = Variable('volume', dtype=np.float32, initial=0.)
    volumeperp = Variable('volumeperp', dtype=np.float32, initial=0.)

# Kernel to check if the particles are stuck
def stuckParticle(particle, fieldset, time):
    if (particle.prev_lon == particle.lon) and (particle.prev_lat == particle.lat):
        particle.stuck += 1
    # Set the stored values for next iteration.
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat

def geodesic(crs, start, end, steps):

    g = Geod(crs.proj4_init)
    geodesic = np.concatenate([
        np.array(start[::-1])[None],
        np.array(g.npts(start[1], start[0], end[1], end[0], steps - 2)),
        np.array(end[::-1])[None]
        ]).transpose()
    points = crs.transform_points(ccrs.Geodetic(), *geodesic)[:, :2]

    return points

def data_extract(var,points):
    lons = var.longitude; lats = var.latitude
    axis_sec =[]
    d_var =[]
    var_sec = []
    
    for i in range(points.shape[0]):
        a = abs(lats - points[i,1]) +abs(lons-points[i,0])
        ii,ij =np.unravel_index(a.argmin(),a.shape)
        dum = var[:,ii,ij]
        var_sec.append(dum.values)
        axis_sec.append(i)
        d = dum.depth.values

    var_sec = np.array(var_sec)

    d_var = d

    return var_sec, axis_sec, d_var #returns the values we want to plot as z,x,y

def distance(lon1,dlon,lat1,dlat):

	# distance between lat/lon points
	R = 6373.0 * 1000. # in meters
	a = (np.sin(dlat/2))**2 + np.cos(radians(lat1)) * np.cos(radians(lon1)) * (np.sin(dlon/2))**2   
	c = 2 * atan2(np.sqrt(a), np.sqrt(1-a))
	dist = R*c
	return dist

# Check if the particles reach the seafloor
def check_seafloor(start,end,steps) :

    # First, get the depths along the transect
    ds =xr.open_dataset(files[0])
    u = ds.uo
    ccrs_ll = ccrs.PlateCarree()
    points_cross = geodesic(ccrs_ll,start,end,steps)
    theta_sec,axis_theta,d_theta = data_extract(u,points_cross)
    
    # get depth along transect
    d_max = np.zeros(len(axis_theta))

    for i in range(len(axis_theta)) :
        j=0
        while np.isnan(theta_sec[i,j]) == False :
            j+=1
        d_max[i] = d_theta[j]
        
    return d_max

#input in the form startXY = (x,y), steps_vert determines depth levels,
#steps determines xy levels

# Creates the initial position of particles
def depthSquare(startXY, endXY, steps, steps_vert):

    #This makes our depth list where each level is spaced out 10m a part.
    depthList = np.arange(0,2000,steps_vert)
    depthList.tolist()

    #Then we want to ensure we have the amount of particles we requested. This is coming from
    #the fact that we are making a rectangle in the ocean. Because we know how many levels 
    #we want going down, and we know how many particles we want we can solve for the 
    #'levels' we need in the lon/lat directions as depthLevels*lon/latLevels = numParticles

    lonList = np.linspace(startXY[1],endXY[1],steps)
    latList = np.linspace(startXY[0],endXY[0],steps)

    #Make empty lists we fill up later

    newLon = []
    newLat = []
    newDepth = []
    area = []

    #Idea of this loop: For each depth level  we fill up our newDepth list with enough copies 
    #of that depth level to match each entry in the lonList,latList

    # Get the depth of the seafloor

#    seafloor = check_seafloor(startXY, endXY, steps)
#
    for i in range(0,len(lonList)):

	# Distance to neighbouring points
        if i==0:
            dlon = abs(lonList[i+1]-lonList[i])/2.
            dlat = abs(latList[i+1]-latList[i])/2.
        elif i==len(lonList):
            dlon = abs(lonList[i]-lonList[i-1])/2.
            dlat = abs(latList[i]-latList[i-1])/2.
        dl = distance(lonList[i], dlon, latList[i], dlat)

        c=0
        for d in depthList:
#            if seafloor[i] > d : # if above seafloor
                newDepth.append(d)
                newLon.append(lonList[i])
                newLat.append(latList[i])
		# Calculate dz for the area of the cell
                if c==0 :
                    dz2 = depthList[c]
                    dz1 = (depthList[c+1]-depthList[c])/2 
                elif c==len(depthList)-1:  
                    dz1 = depthList[c]-depthList[c-1]
                    dz2 = (depthList[c]-depthList[c-1])/2
                else:
                    dz1 = (depthList[c+1]-depthList[c])/2
                    dz2 = (depthList[c]-depthList[c-1])/2
                dz = dz1+dz2
##                area.append( 1./steps*111*1000 * dz )
                area.append( dl * dz )
                c+=1

    return newLon, newLat, newDepth, area

# Check all initial particles to see if S < 34.8 in the climatology, to keep only Labrador Current waters
def CheckWM(Lon, Lat, Depth, Area) :

 # Load the climatology
 dsclim = xr.open_dataset('/storage/mathilde/MainProject/1_ExternalProcesses/ProjectLabCurrent/glorys_S_clim.nc')

 Lon2 = [] ; Lat2 = [] ; Depth2 = [] ; Area2 = []
 for i in range(len(Lon)):
  near = dsclim.so.sel(depth=Depth[i], latitude=Lat[i], longitude=Lon[i], method='nearest')
  if near > 34.8 :
   # do not keep
   idx = i
  else :
   Lon2.append(Lon[i]) ; Lat2.append(Lat[i]) ; Depth2.append(Depth[i]) ; Area2.append(Area[i])

 return Lon2, Lat2, Depth2, Area2

# Kernel to use vertical velocity
def Modified_AdvectionRK4_3D(particle, fieldset, time):
    # make sure we don't exceed the surface and the boundaries
    depmin = .494025 

    if particle.depth <depmin:
        particle.delete()

    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
   
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    if dep1 <depmin :
        particle.delete()

    # from here  = ???    
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]

    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    if dep2 < depmin:
        particle.delete()
    
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]

    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    if dep3 < depmin:
        particle.delete()
          
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]

    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt

    if particle.lon > -30.5 or particle.lat <30.5 or particle.lat > 64.5 or particle.lon < -69.5:
        particle.delete()

    if particle.depth <depmin:
        particle.delete()


# Kernel to track water properties such as temperature and salinity
def SampleVars(particle, fieldset, time):

    particle.temperature = fieldset.T[time, particle.depth, particle.lat, particle.lon]
    particle.salinity = fieldset.S[time, particle.depth, particle.lat, particle.lon]
    (u, v, w) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    particle.u = u
    particle.v = v
    particle.w = w

# Kernel to keep in memory the initial volume associated with each particle
# The initial velocities are zero... so we can't compute the inital volume at the first time step, and we need to store the volume over the whole time series.
def VolumeTrans(particle, fieldset, time):

    if particle.age <= 1:

        (u, v, w) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]

        particle.volume = (u**2+v**2)**0.5 * particle.area
        particle.volumeperp = particle.volume * cos(fieldset.angle + atan(u/v)) 


# Initalization

steps_per_deg = 12 #number of steps per degree (the model as 12 grid points per degree)
steps_vert = 10 # vertical spacing in m

path = '/storage3/shared/Glorys12/withw/'
files = listdir(path)
files.sort()
files = files[:-2]

savename = 'run_continuous_25years_track'

start = (53,-56.7); end = (54.3,-52.0)
ang = atan( (start[0]-end[0])/(start[1]-end[1]) )

steps_hor = int((np.abs(start[1]-end[1]))*steps_per_deg)

Lon, Lat, Depth, Area = depthSquare(start,end,steps_hor,steps_vert)
# Keep only S < 34.8
Lon, Lat, Depth, Area = CheckWM(Lon, Lat, Depth, Area)

# remove particles that would be below the bathymetry
# ALREADY DONE IN CHECK_SEAFLOOR
print('Get initial particles location')
bathy = xr.open_dataset(path[:-6]+'GLO-MFC_001_030_mask_bathy.nc')

Lon2 = [] ; Lat2 = [] ; Depth2 = [] ; Area2 = []
for i in range(len(Lon)):
 if Depth[i] < bathy.deptho.sel(latitude=Lat[i], longitude=Lon[i], method='nearest') :
  Lon2.append(Lon[i])
  Lat2.append(Lat[i])
  Depth2.append(Depth[i])
  Area2.append(Area[i])

Depth = Depth2 ; Lon = Lon2 ; Lat = Lat2 ; Area = Area2
del Depth2, Lon2, Lat2, Area2


# Loop every year
yrs = range(1993,2019)
#yrs = range(2013,2019)
#yrs = range(2013,2015)
for i in range(len(yrs)-3):

	print('Doing loop ', i)

	# --- Get the data --- #
	# Adjust files to the length of the run
	filesl = [path+each for each in files if int(each[30:34])<=yrs[i+3] and int(each[30:34])>=yrs[i]]

	# Get an array of arrays with the time
	# We need to do this because parcels can't read datetime64 data directly
	print('Loading time...')
	times = []
	for f in filesl: 
	    ds = xr.open_dataset(f)
	    times.append( [ds.time.values] )
	print('Done')

	#Setting up the fieldset we will create using our data
	file_names ={'U':filesl,
	             'V':filesl,
	             'W':filesl}
	variables = {'U':'uo',
	             'V':'vo',
	             'W':'w',
                     'T':'thetao',
                     'S':'so'}
	dimensions =  {'lat':'latitude',
	               'lon':'longitude',
	               'depth':'depth'}

	# We give the time with timestamps instead of using the dimension time
	print('Create particle set')

	fset = FieldSet.from_netcdf(filesl,variables,dimensions,timestamps=times)
	fset.add_constant('angle',ang) 

	print('%i particles'%len(Depth))

	# set initialization
	print('Set up initialization')
	repeatdt = timedelta(weeks=1) # release a new set of particles every week

	pset = ParticleSet.from_list(fieldset = fset,
                             pclass = ocean_particle,
                             lat = Lat,
                             lon = Lon,
                             depth = Depth,
                             area = Area,
                             repeatdt = repeatdt
                             )

	#pset.show()

	# Kernel
	k_Modified_AdvectionRK4_3D = pset.Kernel(Modified_AdvectionRK4_3D) + pset.Kernel(ageing) + pset.Kernel(killold) + pset.Kernel(stuckParticle) + pset.Kernel(SampleVars) + pset.Kernel(VolumeTrans)

	#Want to run simulation for 1 years releasing particles every week, and let it run for 3 years.

	output_file = pset.ParticleFile(r"/storage/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Retroflection/%s_%i"%(savename,yrs[i]), outputdt=timedelta(days=1))

	#Start run for one year
	print('First part of run')
	pset.execute(k_Modified_AdvectionRK4_3D,
		runtime = delta(days = 365),  
		dt = delta(minutes = 10),
		output_file=output_file
		)

	#Now we want to stop releasing new particles
	pset.repeatdt = None

	#Now we do the next 3 years with no new particles being released
	print('Second part of run')
	pset.execute(k_Modified_AdvectionRK4_3D,
		runtime = delta(days = 3*365), 
		dt = delta(minutes = 10),
		output_file=output_file
		)

	# Save output
	print('Save')
	output_file.export()

