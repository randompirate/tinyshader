import numpy as np
from PIL import Image
# import Noise
import math
import random as rng
import np_noise


# rng.seed(1)

vs = np.vstack
pi = math.pi

# noise = lambda xx, yy : np.array([Noise.SimplexNoise().noise2(x,y)/2 + 1/2
#                                   for x,y in zip(xx, yy)])

noise = lambda x, y : np_noise.SimplexNoise().noise2(x,y)/2 + 1/2
noise3= lambda x, y, z: np_noise.SimplexNoise().noise3(x,y,z)/2 + 1/2


def fbm(x,y):
  ret = 0
  ret += 1.000*noise( 1*x ,  1*y)
  ret += 0.500*noise( 2*x ,  2*y)
  ret += 0.250*noise( 4*x ,  4*y)
  ret += 0.125*noise( 8*x ,  8*y)
  ret += 0.063*noise(16*x , 16*y)
  ret /= 1.938

  return ret

def star_length(x,y):
  return (np.abs(x) + np.abs(y))

def mix(v1, v2, weight):
  weight = np.clip(weight, 0, 1)
  return (1-weight)*v1 + weight*v2

def smoothstep(minval, maxval, x):
  """ maps a value within [minval, maxval] onto an interval [0,1]
  """
  linear_step = (x - minval) / (maxval - minval)
  linear_step = np.clip(linear_step, 0, 1)

  return 1/2 -  1/2 * np.cos(linear_step * pi)
  # return linear_step

def planet_blur(x,y):
  return (2*planet(x, y) + planet(x+.01, y)+ planet(x, y+.01))/4

def planet(x,y):
  """ x, y are in [-1,1]
      r,g,b are in [0,1]
  """

  # Create relevant functions and constants
  v0 = 0*x
  v1 = v0 + 1
  clr = lambda r,g,b: vs([r*v1, g*v1, b*v1])

  r = np.sqrt(x*x + y*y)        # in [0,1]
  a = (np.arctan2(y, x) + pi) / (2*pi) # in [0,1]

  #Image parameters:
  planet_rad = .6
  main_clrs = [clr(1,0,0), clr(0,0,1)]
  main_noise_f = [.8, 8]

  pale_clr = clr(.5,1,1)
  pale_noise_f = [.2, 8]

  do_ring = True
  ring_radius = .8
  ring_width = .5
  ring_clrs = [clr(0,1,0), clr(.2,.2,1)]

  star_clrs = [clr( 1, 1, 1), clr(.8,.8, 1), clr( 1,.9,.9)]

  do_random = False

  if do_random:
    # complete randomisation
    main_clrs = [clr(rng.random(),rng.random(),rng.random()), clr(rng.random(),rng.random(),rng.random())]
    ring_clrs = [clr(rng.random(),rng.random(),rng.random()), clr(rng.random(),rng.random(),rng.random())]
    pale_clr = clr(rng.random(),rng.random(),rng.random())
    main_noise_f = [.8*rng.random()+.4, 8*rng.random()+4]
    pale_noise_f = [.2*rng.random()+.1, 8*rng.random()+4]
    star_clrs = [clr(.8+rng.random(),.8+rng.random(),.8+rng.random()),
                 clr(.8+rng.random(),.8+rng.random(),.8+rng.random()),
                 clr(.8+rng.random(),.8+rng.random(),.8+rng.random())]


  # Textures
  col = mix(main_clrs[0], main_clrs[1], noise(main_noise_f[0]*x ,main_noise_f[1]*y)) #large scale
  col = mix(col, pale_clr, .6*smoothstep(.3,.6, -.25 + noise(pale_noise_f[0]*x ,pale_noise_f[1]*y))) # Whitewash

  # Planet shape
  space = clr(0,0,0)
  col = mix(space, col, 1-smoothstep(planet_rad - .03, planet_rad  + .03, r))


  # sphere 3D effect
  col = mix(col, clr(0,0,0), .8*smoothstep(.2,1.5*planet_rad,r) )


  # Sun light direction
  col = mix(col, clr(0,0,0), .7*smoothstep(0,1, -x + .5))

  # rings
  if do_ring:
    x_s = 1.0*x - 1.5*y
    y_s = 0.0*x + 3.5*y
    r_s = np.sqrt(x_s*x_s + y_s*y_s)
    a_s = np.arctan2(y_s, x_s)
    r_s = (r_s - ring_radius)/ring_width

    ring = mix(ring_clrs[0], ring_clrs[1], noise(7*r_s,0.01*a_s))
    col = mix(col, ring, ((r > planet_rad) + (y_s>0)) * smoothstep(0,.6, 4*r_s*(1-r_s)))

  # stars
  col = mix(col, star_clrs[0], 1/( 50*star_length(x-.7, y-.8)+.001))
  col = mix(col, star_clrs[1], 1/(100*star_length(x+.2, y-.7)+.001))
  col = mix(col, star_clrs[2], 1/( 75*star_length(x+.3, y+.8)+.001))

  return col


def tunnel(x,y):
  """ x, y are in [-1,1]
      r,g,b are in [0,1]
  """

  # Create relevant functions and constants
  v0 = 0*x
  v1 = v0 + 1
  clr = lambda r,g,b: vs([r*v1, g*v1, b*v1])

  r = np.sqrt(x*x + y*y)        # in [0,1]
  a = (np.arctan2(y, x) + pi) / (2*pi) # in [0,1]

  x_t = y/(x+.5) / (r+.1)
  y_t = x/(y+.5) / (r)
  col = mix(clr(0,0,0), clr(1,0,1) , fbm(12*x_t,y_t))



  return col

def test(x,y):

  # Create relevant functions and constants
  v0 = 0*x
  v1 = v0 + 1
  clr = lambda r,g,b: vs([r*v1, g*v1, b*v1])

  # Polar coords
  r = np.sqrt(x*x + y*y)
  a = np.arctan2(-y, -x) + pi
  ga = 137.508 /180 *pi

  return mix(clr(0,0,0), clr(1,1,1), ( np.minimum((3 * r + a)%ga, (3 * r + a - 2*pi)%ga) < .1)*1 )




def render(shaderfun, imname = 'render.png'):
  # Set up rgb array
  width  = 400
  height = 400

  # point array in [-1,1]^2
  xx =  (2 * np.arange(width)  - width  + 1) / (width  - 1)    # (1) x (width)
  yy = -(2 * np.arange(height) - height + 1) / (height - 1)    # (1) x (height)
  PP = np.array([np.tile(xx, height), np.repeat(yy, width)])  # (1) x (width*height)

  # Call shaderfunction for RGB vals
  CC = shaderfun(PP[0,:], PP[1,:])  # (3) x (width*height)

  # Clip and map to {0,1, .. 254}. Create imdata list
  CC = (255* np.clip(CC, 0, 1)).astype(int)
  imdata = [tuple(rgb) for rgb in np.transpose(CC).tolist()]

  # Write image
  newimage = Image.new('RGB', (width, height))
  newimage.putdata(imdata)
  newimage.save(imname)  # takes type from filename extension

render(planet)