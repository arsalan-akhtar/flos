--[[
Example on how to use an NEB method.
--]]

-- Load the FLOS module
local flos = require "flos"

-- The prefix of the files that contain the images
local image_label = "GEOMETRY_"

-- Total number of images (excluding initial[0] and final[n_images+1])
local n_images = 3
-- Table of image geometries
local images = {}

-- The default output label of the DM files
local label = "siesta"

-- Function for reading a geometry
local read_geom = function(filename)
   local file = io.open(filename, "r")
   local na = tonumber(file:read())
   local R = flos.Array.zeros(na, 3)
   file:read()
   local i = 0
   local function tovector(s)
      local t = {}
      s:gsub('%S+', function(n) t[#t+1] = tonumber(n) end)
      return t
   end
   for i = 1, na do
      local line = file:read()
      if line == nil then break end
      -- Get stuff into the R
      local v = tovector(line)
      R[i][1] = v[1]
      R[i][2] = v[2]
      R[i][3] = v[3]
   end
   file:close()
   return R
end

-- Now read in the images
for i = 0, n_images + 1 do
   images[#images+1] = flos.MDStep:new{R=read_geom(image_label .. i .. ".xyz")}
end

-- Now we have all images...
local NEB = flos.NEB:new(images)
-- Remove global (we use NEB.n_images)
n_images = nil

-- Setup each image relaxation method (note it is prepared for several
-- relaxation methods per-image)
local relax = {}
for i = 1, NEB.n_images do
   -- Select the relaxation method
   relax[i] = {}
   relax[i][1] = flos.LBFGS:new({H0 = 1. / 75})
   --relax[i][2] = flos.LBFGS:new({H0 = 1. / 50})

   --relax[i][1] = flos.FIRE:new({dt_init = 1., direction="global", correct="global"})
   -- add more relaxation schemes if needed ;)
end

-- Counter for controlling which image we are currently relaxing
local current_image = 1

-- Grab the unit table of siesta (it is already created
-- by SIESTA)
local Unit = siesta.Units


function siesta_comm()
   
   -- This routine does exchange of data with SIESTA
   local ret_tbl = {}

   -- Do the actual communication with SIESTA
   if siesta.state == siesta.INITIALIZE then
      
      -- In the initialization step we request the
      -- convergence criteria
      --  MD.MaxDispl
      --  MD.MaxForceTol
      siesta_get({"Label",
		  "geom.xa",
		  "MD.MaxDispl",
		  "MD.MaxForceTol"})

      -- Store the Label
      label = tostring(siesta.Label)

      -- Print information
      IOprint("\nLUA NEB calculator")

      -- Ensure we update the convergence criteria
      -- from SIESTA (in this way one can ensure siesta options)
      for img = 1, NEB.n_images do
	 if siesta.IONode then
	    print(("\nLUA NEB relaxation method for image %d:"):format(img))
	 end
	 for i = 1, #relax[img] do
	    relax[img][i].tolerance = siesta.MD.MaxForceTol * Unit.Ang / Unit.eV
	    relax[img][i].max_dF = siesta.MD.MaxDispl / Unit.Ang
	    
	    -- Print information for this relaxation method
	    if siesta.IONode then
	       relax[img][i]:info()
	    end
	 end
      end

      -- This is only reached one time, and that it as the beginning...
      -- be sure to set the corresponding values
      siesta.geom.xa = NEB.initial.R * Unit.Ang

      IOprint("\nLUA/NEB initial state\n")

      -- force the initial image to be the first one to run
      current_image = 0

      ret_tbl = {'geom.xa'}

   end

   if siesta.state == siesta.MOVE then
      
      -- Here we are doing the actual LBFGS algorithm.
      -- We retrieve the current coordinates, the forces
      -- and whether the geometry has relaxed
      siesta_get({"geom.fa",
		  "E.total",
		  "MD.Relaxed"})

      -- Store the old image that has been tested,
      -- in this way we can check whether we have moved to
      -- a new image.
      local old_image = current_image
      
      ret_tbl = siesta_move(siesta)

      -- we need to re-organize the DM files for faster convergence
      -- pass whether the image is the same
      siesta_update_DM(old_image, current_image)

   end

   siesta_return(ret_tbl)
end

function siesta_move(siesta)

   -- Retrieve the atomic coordinates, forces and the energy
   local fa = flos.Array.from(siesta.geom.fa) * Unit.Ang / Unit.eV
   local E = siesta.E.total / Unit.eV

   -- First update the coordinates, forces and energy for the
   -- just calculated image
   NEB[current_image]:set{F=fa, E=E}

   if current_image == 0 then
      -- Perform the final image, to retain that information
      current_image = NEB.n_images + 1

      -- Set the atomic coordinates for the final image
      siesta.geom.xa = NEB[current_image].R * Unit.Ang

      IOprint("\nLUA/NEB final state\n")

      -- The siesta relaxation is already not set
      return {'geom.xa'}
      
   elseif current_image == NEB.n_images + 1 then

      -- Start the NEB calculation
      current_image = 1

      -- Set the atomic coordinates for the final image
      siesta.geom.xa = NEB[current_image].R * Unit.Ang

      IOprint(("\nLUA/NEB running NEB image %d / %d\n"):format(current_image, NEB.n_images))
	 
      -- The siesta relaxation is already not set
      return {'geom.xa'}

   elseif current_image < NEB.n_images then

      -- Figure out the next image
      current_image = current_image + 1
      while relax[current_image][1]:optimized() do
	 current_image = current_image + 1
	 
	 if current_image > NEB.n_images then
	    break
	 end
      end
      
      if current_image <= NEB.n_images then
	 
	 -- Set the atomic coordinates for the image
	 siesta.geom.xa = NEB[current_image].R * Unit.Ang
	 
	 IOprint(("\nLUA/NEB running NEB image %d / %d\n"):format(current_image, NEB.n_images))
	 
	 -- The siesta relaxation is already not set
	 return {'geom.xa'}

      else
	 -- The NEB routine have the remaining images
	 -- relaxed, so we proceed with the NEB-force method
      end

   end
   
   -- First we figure out how perform the NEB optimizations
   -- Now we have calculated all the systems and are ready for doing
   -- an NEB MD step

   -- Global variable to check for the NEB convergence
   -- Initially assume it has relaxed
   local relaxed = true

   IOprint("\nNEB step")
   local out_R = {}

   -- loop on all images and pass the updated forces to the mixing algorithm
   for img = 1, NEB.n_images do

      -- Get the correct NEB force (note that the relaxation
      -- methods require the negative force)
      local F = NEB:force(img, siesta.IONode)
      IOprint("NEB: max F on image ".. img .. (" = %10.5f"):format(F:norm():max()) )

      -- Prepare the relaxation for image `img`
      local all_xa, weight = {}, flos.Array( #relax[img] )
      for i = 1, #relax[img] do
	 all_xa[i] = relax[img][i]:optimize(NEB[img].R, F)
	 weight[i] = relax[img][i].weight
      end
      weight = weight / weight:sum()

      if siesta.IONode and #relax[img] > 1 then
	 print("\n weighted average for relaxation: ", tostring(weight))
      end
      
      -- Calculate the new coordinates and figure out
      -- if the algorithms has been optimized.
      local out_xa = all_xa[1] * weight[1]
      relaxed = relaxed and relax[img][1]:optimized()
      for i = 2, #relax[img] do
	 out_xa = out_xa + all_xa[i] * weight[i]
	 relaxed = relaxed and relax[img][i]:optimized()
      end
      
      -- Copy the optimized coordinates to a table
      out_R[img] = out_xa

   end

   -- Before we update the coordinates we will write
   -- the current steps results to the result file
   -- (this HAS to be done before updating the coordinates)
   NEB:save( siesta.IONode )

   -- Now we may copy over the coordinates (otherwise
   -- we do a consecutive update, and then overwrite)
   for img = 1, NEB.n_images do
      NEB[img]:set{R=out_R[img]}
   end
   
   -- Start over in case the system has not relaxed
   current_image = 1
   if relaxed then
      -- the final coordinates are returned
      siesta.geom.xa = NEB.final.R * Unit.Ang
      IOprint("\nLUA/NEB complete\n")
   else
      siesta.geom.xa = NEB[1].R * Unit.Ang
      IOprint(("\nLUA/NEB running NEB image %d / %d\n"):format(current_image, NEB.n_images))
   end

   siesta.MD.Relaxed = relaxed
      
   return {"geom.xa",
	   "MD.Relaxed"}
end

function file_exists(name)
   local f = io.open(name, "r")
   if f ~= nil then
      io.close(f)
      return true
   else
      return false
   end
end

-- Function for retaining the DM files for the images so that we
-- can easily restart etc.
function siesta_update_DM(old, current)

   if not siesta.IONode then
      -- only allow the IOnode to perform stuff...
      return
   end

   -- Move about files so that we re-use old DM files
   local DM = label .. ".DM"
   local old_DM = DM .. "." .. tostring(old)
   local current_DM = DM .. "." .. tostring(current)

   if 1 <= old and old <= NEB.n_images and file_exists(DM) then
      -- store the current DM for restart purposes
      IOprint("Saving " .. DM .. " to " .. old_DM)
      os.execute("mv " .. DM .. " " .. old_DM)
   elseif file_exists(DM) then
      IOprint("Deleting " .. DM .. " for a clean restart...")
      os.execute("rm " .. DM)
   end

   if file_exists(current_DM) then
      IOprint("Restoring " .. current_DM .. " to " .. DM)
      os.execute("cp " .. current_DM .. " " .. DM)
   end

end
