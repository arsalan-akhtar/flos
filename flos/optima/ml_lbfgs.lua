---
-- Implementation of the limited memory BFGS algorithm
-- @classmod ML_LBFGS

local m = require "math"
local mc = require "flos.middleclass.middleclass"

local num = require "flos.num"
local optim = require "flos.optima.base"

-- Create the ML_LBFGS class (inheriting the Optimizer construct)
local ML_LBFGS = mc.class("ML_LBFGS", optim.Optimizer)


--- Instantiating a new `ML_LBFGS` object.
--
-- The ML_LBFGS algorithm is a straight-forward optimization algorithm which requires
-- very few arguments for a succesful optimization.
-- The most important parameter is the initial Hessian value, which for large values (close to 1)
-- may have difficulties in converging because it is more aggressive (keeps more of the initial
-- gradient). The default value is rather safe and should enable optimization on most systems.
--
-- This optimization method also implements a history-discard strategy, if needed, for possible
-- speeding up the convergence. A field in the argument table, `discard`, may be passed which
-- takes one of
--  - "none", no discard strategy
--  - "max-dF", if a displacement is being made beyond the max-displacement we do not store the
--   step in the history
--
-- This optimization method also implements a scaling strategy, if needed, for possible speeding
-- up the convergence. A field in the argument table, `scaling`, may be passed which takes one of
--  - "none", no scaling strategy used
--  - "initial", only the initial inverse Hessian and use that in all subsequent iterations
--  - "every", scale for every step
--
-- @usage
-- ML_LBFGS = ML_LBFGS{<field1 = value>, <field2 = value>}
-- while not lbfgs:optimized() do
--    F = lbfgs:optimize(F, G)
-- end
--
-- @function ML_LBFGS:new
-- @number[opt=1] damping damping parameter for the parameter change
-- @number[opt=1/75] H0 initial Hessian value, larger values are more safe, but takes possibly longer to converge
-- @int[opt=25] history number of previous steps used when calculating the new Hessian
-- @string[opt="none"] discard method for discarding a previous history step
-- @string[opt="none"] scaling method for scaling the inverse Hessian
-- @param ... any arguments `Optimizer:new` accepts
local function doc_function()
end

function ML_LBFGS:initialize(tbl)
   if siesta.IONode then
   ML_LBFGS:PrintBetweenCenteredWithBorders("RELAXATION HYBRID BA ML","-")
   end 
   -- Initialize from generic optimizer
   optim.Optimizer.initialize(self)

   local tbl = tbl or {}

   -- Damping of the BFGS algorithm
   --  damping > 1
   --    over-relaxed
   --  damping < 1
   --    under-relaxed
   self.damping = 1.0
   
   -- Initial inverse Hessian
   -- Lower values converges faster at the risk of
   -- instabilities
   -- Larger values are easier to converge
   self.H0 = 1. / 75.
   self.H0_init = 1. / 75.

   -- Scaling method for the initial Hessian
   --  - none
   --  - initial (dG . dF / |dG|^2)
   --  - every (dG . dF / |dG|^2)
   self.scaling = "none"

   -- Number of previous history points used
   self.history = 25
   -- The discard method for the history step
   --  - none
   --  - max-dF
   self.discard = "none"

   -- Field of the functional we wish to optimize
   --
   --   F == optimization variable/functional
   --   G == gradient variable/functional (minimization)
   self.F0 = nil
   self.G0 = nil

   -- History fields of the residuals.
   -- We store the residuals of the
   --   dF == optimization variable/functional
   --   dG == gradient variable/functional (minimization)
   --   rho is the kernel of the residual dot-product
   self.dF = {}
   self.dG = {}
   self.rho = {}
   -- The last G . dF using dF for the optimized step
   self.weight = 1.

   -- Ensure we update the elements as passed
   -- by new(...)
   if type(tbl) == "table" then
      for k, v in pairs(tbl) do
	 self[k] = v
      end
   end

   -- Ensure the initial H0 is "fixed"
   self.H0_init = self.H0

end

--- Reset the `ML_LBFGS` object
function ML_LBFGS:reset()
   optim.Optimizer.reset(self)
   -- Copy over the initial H0 (for safety)
   self.H0 = self.H0_init
   self.F0 = nil
   self.G0 = nil
   self.dF = {}
   self.dG = {}
   self.rho = {}
   self.weight = 1.
end



--- Normalize the parameter displacement to a given max-change.
-- The ML_LBFGS algorithm always perfoms a global correction to maintain
-- the minimization direction.
-- @Array dF the parameter displacements that are to be normalized
-- @return the normalized `dF` according to the `global` or `local` correction
function ML_LBFGS:correct_dF(dF)

   -- Calculate the maximum norm
   local max_norm
   if #dF.shape == 1 then
      max_norm = dF:abs():max()
   else
      max_norm = dF:norm():max()
   end

   -- Now normalize the displacement
   local norm = self.max_dF / max_norm
   if norm < 1. then
      return dF * norm
   else
      return dF
   end
   
end

--- Add the current optimization variable and the gradient variable to the history.
-- This function calculates the residuals and updates the kernel of the residual dot-product.
-- @Array F the parameters for the function
-- @Array G the gradient of the function with the parameters `F`
function ML_LBFGS:add_history(F, G)

   -- Retrieve the current iteration step.
   -- With respect to the history and total
   -- iteration count.
   local iter = self:_history()

   -- If the current iteration count is
   -- more than or equal to one, it means that
   -- we already have F0 and G0
   if self.F0 ~= nil then

      -- Increase history 
      iter = iter + 1

      self.dF[iter] = F - self.F0
      self.dG[iter] = G - self.G0
      
      -- Calculate dot-product and store the kernel
      self.rho[iter] = -1. / self.dF[iter]:flatdot(self.dG[iter])
      if self.rho[iter] == -m.huge or m.huge == self.rho[iter] then
	 -- An inf number 
	 self.rho[iter] = 0.
      elseif self.rho[iter] ~= self.rho[iter] then
	 -- A nan number does not equal it-self
	 self.rho[iter] = 0.
      end

   end
   
   -- In case we have stored too many points
   -- we should clean-up the history
   if iter > self.history then

      self:remove_history()
	 
   end

   -- Ensure that the next iteration has
   -- the input sequence
   self.F0 = F:copy()
   self.G0 = G:copy()

end

--- Removes an element from the history
-- @int[opt=1] index the index of the history to remove (1 == oldest)
-- @local
function ML_LBFGS:remove_history(index)
   local idx = index or 1
   if idx > #self.dF then
      return
   end
   
   -- Remove history stuff...
   -- This will automatically reorder the table
   table.remove(self.dF, idx)
   table.remove(self.dG, idx)
   table.remove(self.rho, idx)

end


--- Return the current number of histories saved
-- @return number of stored iterations
-- @local
function ML_LBFGS:_history()
   -- This is simply the number of elements in the dF
   if self.F0 == nil then
      return 0
   else
      return #self.dF
   end
end


--- Perform a ML_LBFGS step with input parameters `F` and gradient `G`
-- @Array F the parameters for the function
-- @Array G the gradient for the function with parameters `F`
-- @return a new set of parameters which should converge towards a
--   local minimum point.
function ML_LBFGS:optimize(F, G)
   
   -- Add the current iteration to the history
   self:add_history(F, G)

   -- Retrieve current number of previous elements stored
   local iter = self:_history()

   -- Create local pointers to tables
   -- (they are tables, hence by-reference)
   local dF = self.dF
   local dG = self.dG
   local rho = self.rho

   -- Create table for accumulating dot products
   local rh = {}
   
   -- Update the downhill gradient
   local q = - G:flatten()
   for i = iter, 1, -1 do
      rh[i] = rho[i] * dF[i]:flatdot(q)
      q = q + rh[i] * dG[i]:flatten()
   end

   -- Solve for the rhs optimization
   local z
   if self.scaling == "initial" then
      if iter == 1 then
	 self.H0 = self.H0_init * dG[iter]:flatdot(dF[iter]) /
	    dG[iter]:flatdot(dG[iter])
      end
      z = q * self.H0
   elseif self.scaling == "every" and iter > 0 then
      z = q * self.H0 * dG[iter]:flatdot(dF[iter]) /
	 dG[iter]:flatdot(dG[iter])
   else
      z = q * self.H0
   end
   -- Clean-up
   q = nil

   -- Now create the next step
   for i = 1, iter do
      local beta = rho[i] * dG[i]:flatdot(z)
      z = z + dF[i]:flatten() * (rh[i] + beta)
   end
   
   -- Ensure shape
   z = - z:reshape(G.shape)
   
   -- Update step
   self.weight = m.abs(G:flatdot(z))
   local dF = self:correct_dF(z)

   -- Figure out if we should discard some of the previous steps...
   if self.discard == "max-dF" and (dF - z):sum(0) ~= 0. and iter > 0 then
      print("LUA removed history step "..tostring(iter))
      self:remove_history(iter)
   end
   
   -- Determine whether we have optimized the parameter/functional
   self:optimized(G)
   
   self.niter = self.niter + 1

   -- return optimized coordinates, regardless
   return F + dF * self.damping
      
end


--- SIESTA function for performing a complete SIESTA ML_LBFGS optimization.
--
-- This function will query these fdf-flags from SIESTA:
--
--  - MD.MaxForceTol
--  - MD.MaxCGDispl
--
-- and use those as the tolerance for convergence as well as the
-- maximum displacement for each optimization step.
--
-- Everything else is controlled by the `ML_LBFGS` object.
--
-- Note that all internal operations in this function relies on units being in
--  - Ang
--  - eV
--  - eV/Ang
--

-- @tparam table siesta the SIESTA global table.
-- function ML_LBFGS:SIESTA(siesta)
   -- local unit = siesta.Units

   -- if siesta.state == siesta.INITIALIZE then
   --    if self.relaxation_counter == nil then
   --       self.relaxation_counter = 0
   --    end
   --    if siesta.IONode then
   --       ML_LBFGS:PrintBetweenCenteredWithBorders("AA:INJA ","-")
   --       ML_LBFGS:check_file_exists("siestayaml.yml")
   --    end
   --    siesta.receive({"Label","MD.MaxDispl","geom.xa", "MD.MaxForceTol"})
   --    local xa_init = num.Array.from(siesta.geom.xa) / unit.Ang
   --    if siesta.IONode then
   --       print("xa_init:")
   --       print(xa_init)
   --       print("LABELE SIESTA HAST: "..siesta.Label)
   --    end
   --    self.tolerance = siesta.MD.MaxForceTol * unit.Ang / unit.eV
   --    self.max_dF = siesta.MD.MaxDispl / unit.Ang

   --    if siesta.IONode then
   --       self:info()
   --    end

   -- elseif siesta.state == siesta.MOVE then
   --    if siesta.IONode then
   --       ML_LBFGS:PrintBetweenCenteredWithBorders("AA:INJA DAR HALATE SIESTA MOVE ","-")
   --       ML_LBFGS:check_file_exists("siestayaml.yml")
   --    end
   --    self.relaxation_counter = self.relaxation_counter + 1
   --    if siesta.IONode then
   --       print("Relaxation Counter: ".. self.relaxation_counter)
   --    end

   --    if self.relaxation_counter == 1 then
   --       local output_prefix = "siesta_ml_" .. self.relaxation_counter
   --       local command = "mldft siesta-relax siestayaml.yml -w " .. output_prefix
   --       local poscar_data
   --       if siesta.IONode then
   --          print("Root process (IONode) running command: " .. command)
   --          local success = self:run_command(command)
   --          if not success then
   --             error("Command execution failed: " .. command)
   --          end
   --          local status, result = pcall(function()
   --             return ML_LBFGS:read_poscar_to_flos_array({subdirectory = output_prefix, filename = "SIESTA_MLRELAXER_driect.out"})
   --          end)
   --          if not status or not result then
   --             error("Error reading POSCAR: " .. tostring(result or "unknown error"))
   --          end
   --          poscar_data = result
   --          print("POSCAR COORDINATES:")
   --          print(poscar_data.coord)
   --          siesta.geom.xa = poscar_data.coord * unit.Ang
   --       end
   --       -- Distribute geom.xa to all processes
   --       siesta.send({'geom.xa'})
   --       siesta.receive({'geom.xa'})
   --       if siesta.IONode then
   --          print("After send/receive, geom.xa:", siesta.geom.xa)
   --       end
   --    end

   --    siesta.receive({"geom.xa", "geom.fa", "MD.Relaxed"})
   --    if siesta.IONode then
   --       print("Received geom.xa:", siesta.geom.xa)
   --       print("Received geom.fa:", siesta.geom.fa)
   --    end
   --    if self.relaxation_counter > 1 then
   --       local xa = num.Array.from(siesta.geom.xa) / unit.Ang
   --       local fa = num.Array.from(siesta.geom.fa) * unit.Ang / unit.eV
   --       if siesta.IONode then
   --          print("xa:", xa)
   --          print("fa:", fa)
   --       end
   --       if self:optimized(fa) then
   --          siesta.MD.Relaxed = true
   --          siesta.send({"MD.Relaxed"})
   --       else
   --          siesta.geom.xa = self:optimize(xa, fa) * unit.Ang
   --          siesta.send({"geom.xa"})
   --          siesta.receive({"geom.xa"}) -- Ensure all processes get updated geom.xa
   --          if siesta.IONode then
   --             print("After optimize, geom.xa:", siesta.geom.xa)
   --          end
   --       end
   --    end
   -- end
-- end

function ML_LBFGS:SIESTA(siesta)
   local unit = siesta.Units

   if siesta.state == siesta.INITIALIZE then
      if self.relaxation_counter == nil then
         self.relaxation_counter = 0
      end
      if siesta.IONode then
         ML_LBFGS:PrintBetweenCenteredWithBorders("AA:INJA ","-")
         ML_LBFGS:check_file_exists("siestayaml.yml")
      end
      siesta.receive({"Label","MD.MaxDispl","geom.xa", "MD.MaxForceTol"})
      if siesta.IONode then
         print("xa_init:", siesta.geom.xa)
         print("LABELE SIESTA HAST: "..siesta.Label)
      end
      self.tolerance = siesta.MD.MaxForceTol * unit.Ang / unit.eV
      self.max_dF = siesta.MD.MaxDispl / unit.Ang
      if siesta.IONode then
         self:info()
      end

   end -- siesta.state == siesta.INITIALIZE
   
   if siesta.state == siesta.MOVE then
      -- Receive information
      
      siesta.receive({"geom.xa", "geom.fa", "MD.Relaxed"})
      self.relaxation_counter = self.relaxation_counter + 1
      if siesta.IONode then
         ML_LBFGS:PrintBetweenCenteredWithBorders("AA:INJA DAR HALATE SIESTA MOVE ","-")
         ML_LBFGS:check_file_exists("siestayaml.yml")
         print("Relaxation Counter: "..self.relaxation_counter)
      end

      
      -- siesta.receive({"geom.xa", "geom.fa", "MD.Relaxed"})

      -- Now retrieve the coordinates and the forces
      local xa = num.Array.from(siesta.geom.xa) / unit.Ang
      local fa = num.Array.from(siesta.geom.fa) * unit.Ang / unit.eV

      -- Only in case that the forces are optimized will we move atoms.
      if self:optimized(fa) then
         siesta.MD.Relaxed = true
         siesta.send({"MD.Relaxed"})
      else      
         if self.relaxation_counter == 1 then
            if siesta.IONode then
               print("DAR self.relaxation_counter 1 :" .. self.relaxation_counter)
               local output_prefix = "siesta_ml_" .. self.relaxation_counter
               local command = "mldft siesta-relax siestayaml.yml -w " .. output_prefix
               print("Root process (IONode) running command: " .. command)
               local success = self:run_command(command)
               if not success then
                  error("Command execution failed: " .. command)
               end
               local status, result = pcall(function()
               return ML_LBFGS:read_poscar_to_flos_array({subdirectory = output_prefix, filename = "SIESTA_MLRELAXER_driect.out"})
               end)
               if not status or not result then
                  error("Error reading POSCAR: " .. tostring(result or "unknown error"))
               end
               print("DEBUG: POSCAR COORDINATES:", result.coord * unit.Ang)
               -- print("DEBUG: geom.fa", num.Array.from(siesta.geom.fa ) * unit.Ang / unit.eV)
               -- siesta.geom.xa = result.coord * unit.Ang   
               self.ml_geom = result.coord * unit.Ang               
            end -- siesta.IONode then
               print("DEBUG: ml_geom",self.ml_geom)
               siesta.geom.xa = self.ml_geom
               print("DEBUG: siesta.geom.xa",siesta.geom.xa)
               siesta.send({'geom.xa'})  -- ,"geom.fa"
         else 
               -- Send back new coordinates (convert to Bohr)
               siesta.geom.xa = self:optimize(xa, fa) * unit.Ang
               siesta.send({"geom.xa"})
         end -- self.relaxation_counter == 1             
      end
    
   end -- if siesta.state == siesta.MOVE then
end  -- function ML_LBFGS:SIESTA(siesta)
   








--- Print information regarding the `ML_LBFGS` object
function ML_LBFGS:info()

   print("")
   local it = self:_history()
   if it == 0 then
      print("ML_LBFGS: history: " .. self.history)
   else
      print("ML_LBFGS: current / history: "..tostring(it) .. " / "..self.history)
   end
   print("ML_LBFGS: damping "..tostring(self.damping))
   print("ML_LBFGS: H0 "..tostring(self.H0_init))
   print("ML_LBFGS: scaling "..self.scaling)
   print("ML_LBFGS: discard "..self.discard)
   print("ML_LBFGS: Tolerance "..tostring(self.tolerance))
   print("ML_LBFGS: Maximum change "..tostring(self.max_dF))
   print("")

end

function ML_LBFGS:PrintCenteredWithBorders(text)
  local borderChar = "="
  local totalWidth = 80

  if string.len(text) > totalWidth - 4 then -- Check if text is too long
    print(string.rep(borderChar, totalWidth))
    print("Text too long to center within " .. totalWidth .. " characters.")
    print(string.rep(borderChar, totalWidth))
    return
  end

  local padding = math.floor((totalWidth - string.len(text)) / 2)
  local leftPadding = string.rep(" ", padding)
  local rightPadding = string.rep(" ", totalWidth - string.len(text) - padding) -- Corrected right padding

  local border = string.rep(borderChar, totalWidth)

  print(border)
  print(leftPadding .. text .. rightPadding)
  print(border)
end


function ML_LBFGS:PrintBetweenCenteredWithBorders(text,borderChar, totalWidth)
  -- local borderChar = "="
borderChar = borderChar or "="
totalWidth = totalWidth or 80

  local textWidth = string.len(text)
  local border = string.rep(borderChar, textWidth + 4) -- Border width based on text length
  local padding = math.floor((totalWidth - string.len(border)) / 2)
  local leftPadding = string.rep(" ", padding)
  local rightPadding = string.rep(" ", totalWidth - string.len(border) - padding)

  local textPadding = math.floor((string.len(border) - textWidth)/2)
  local textLeftPadding = string.rep(" ", textPadding)
  local textRightPadding = string.rep(" ", string.len(border) - textWidth - textPadding)

  print(leftPadding .. border .. rightPadding)
  print(leftPadding .. textLeftPadding .. text .. textRightPadding .. rightPadding)
  print(leftPadding .. border .. rightPadding)
end

--- Executes a system command using the operating system shell.
-- Handles compatibility between Lua 5.1/5.2 and Lua 5.3+ regarding os.execute return values.
-- Prints status messages about the execution.
--
-- @param cmd_string (string) The command to execute.
-- @return boolean `true` if the command executed successfully (exit code 0),
--                 `false` otherwise (failed to execute or non-zero exit code).
--
-- @usage
-- local ok = run_command("echo Hello World")
-- if ok then
--     print("Command ran successfully.")
-- else
--     print("Command failed.")
-- end
function ML_LBFGS:run_command(cmd_string)
   if not cmd_string or cmd_string == "" then
     print("Error: No command string provided.")
     return false
   end
 
   print("Attempting to run command: " .. cmd_string)
 
   -- Execute the command using the operating system shell
   -- Lua 5.1/5.2: Returns true (for exit code 0), false (non-zero exit), or nil (cannot execute).
   -- Lua 5.3+: Returns the exit code (0 for success, non-zero for error), or nil (cannot execute).
   local result = os.execute(cmd_string)
 
   -- Check for successful execution (exit code 0) across Lua versions
   if result == true or result == 0 then
     print("Command executed successfully (exit code 0).")
     return true -- Success
   else
     
     if result == nil then
       print("Error: Command failed to execute (e.g., command not found, permission denied).")
     else -- result is false (Lua 5.1/5.2) or a non-zero number (Lua 5.3+)
       local exit_code_str = type(result) == "number" and tostring(result) or "(non-zero)"
       print("Error: Command executed but returned an error status (exit code: " .. exit_code_str .. ").")
     end
     
     return false -- Failure
   end
 end


-- Lua script to copy a file from a subdirectory to the current directory,
-- with an option to specify a new name for the copied file.

-- Function to copy a file from a specified subdirectory to the current directory.
-- Assumes the script is executed from the parent directory of the subdirectory.
-- Allows specifying a different name for the destination file.
-- function ML_LBFGS:copy_file_from_subdirectory(filename, subdirectory, destination_filename)
function ML_LBFGS:copy_file_from_subdirectory(args)
   -- Set default values if arguments are nil or empty
   filename = (args.filename and #args.filename > 0) and args.filename or "SIESTA_MLRELAXER.XV"
   subdirectory = (args.subdirectory and #args.subdirectory > 0) and args.subdirectory or "siesta_ml_1"
   -- If destination_filename is not provided or empty, use the original filename
   destination_filename = (args.destination_filename and #args.destination_filename > 0) and args.destination_filename or args.filename

   -- Construct paths relative to the script's execution directory.
   -- NOTE: This assumes '/' as the path separator, which works on Linux, macOS,
   -- and often on modern Windows. For full Windows compatibility with '\',
   -- more complex path handling or an external library like luafilesystem might be needed.
   local source_path = subdirectory .. "/" .. filename
   -- Use the potentially user-defined destination filename for the path in the current directory
   local destination_path = destination_filename

   print("Attempting to copy '" .. source_path .. "' to '" .. destination_path .. "'")

   -- 1. Check if source file exists and open it for reading (binary mode)
   local source_file, err_open_source = io.open(source_path, "rb")
   if not source_file then
       print("Error: Could not open source file '" .. source_path .. "'.")
       print("Reason: " .. (err_open_source or "File not found or no read permission."))
       return false
   end
   print("Source file opened successfully.")

   -- 2. Open destination file for writing (binary mode)
   -- This will create the file if it doesn't exist, or overwrite it if it does.
   local dest_file, err_open_dest = io.open(destination_path, "wb")
   if not dest_file then
       print("Error: Could not open destination file '" .. destination_path .. "' for writing.")
       print("Reason: " .. (err_open_dest or "No write permission in the current directory?"))
       source_file:close() -- Close the source file before returning
       return false
   end
   print("Destination file opened successfully.")

   -- 3. Copy data in chunks to handle potentially large files
   local success = true
   local chunk_size = 8192 -- 8 KB chunk size (adjust as needed)
   print("Starting file copy...")
   while true do
       -- Read a chunk from the source
       local chunk, err_read = source_file:read(chunk_size)

       -- Handle read errors
       if err_read then
           print("Error: Failed reading from source file '" .. source_path .. "'.")
           print("Reason: " .. err_read)
           success = false
           break
       end

       -- If chunk is nil, we've reached the end of the file
       if not chunk then
           break
       end

       -- Write the chunk to the destination
       local _, err_write = dest_file:write(chunk)

       -- Handle write errors
       if err_write then
           print("Error: Failed writing to destination file '" .. destination_path .. "'.")
           print("Reason: " .. err_write)
           success = false
           break
       end
   end

   -- 4. Close both files
   -- It's good practice to check if closing was successful, though errors here are less common.
   local ok_close_source, err_close_source = source_file:close()
   local ok_close_dest, err_close_dest = dest_file:close()

   if not ok_close_source then
       print("Warning: Error closing source file '" .. source_path .. "': " .. (err_close_source or "Unknown error"))
       -- Consider the copy potentially incomplete if closing failed after write errors
   end
   if not ok_close_dest then
       print("Warning: Error closing destination file '" .. destination_path .. "': " .. (err_close_dest or "Unknown error"))
       -- Data might be flushed, but report the warning.
   end

   -- 5. Report final status and clean up if failed
   if success then
       -- Updated success message to show both source and destination names clearly
       print("File '" .. filename .. "' from '".. subdirectory .."' successfully copied as '" .. destination_path .. "' in the current directory.")
   else
       print("File copy operation failed for source '" .. source_path .. "'.")
       -- Attempt to remove the potentially incomplete or corrupted destination file
       print("Attempting to remove potentially incomplete destination file: " .. destination_path)
       local removed, err_remove = os.remove(destination_path)
       if not removed then
           print("Warning: Could not remove incomplete destination file '" .. destination_path .. "'.")
           print("Reason: " .. (err_remove or "Unknown error"))
       end
   end

   return success
end


-- Lua script to check if a specific file exists in the current directory

-- Function to check if a file exists and is readable in the current directory
-- Returns true if the file exists and is readable, false otherwise.
function ML_LBFGS:check_file_exists(filename)
   -- Input validation: ensure filename is a non-empty string
   if not filename or type(filename) ~= "string" or #filename == 0 then
       print("Error: Invalid filename provided.")
       return false
   end

   print("Attempting to check for file: '" .. filename .. "'")

   -- Attempt to open the file in read mode ("r").
   -- io.open returns a file handle if successful, or nil + error message otherwise.
   -- We don't need to read the file, just see if it can be opened.
   local file, err_msg = io.open(filename, "r")

   if file then
       -- Success! The file exists and we have permission to read it.
       print("Result: File '" .. filename .. "' exists and is accessible in the current directory.")
       -- It's crucial to close the file handle once we're done.
       file:close()
       return true
   else
       -- Failure. The file either doesn't exist or we don't have permission.
       -- The error message might give more details, but for this check,
       -- we'll report it as not found or inaccessible.
       print("Result: File '" .. filename .. "' does not exist or is not accessible.")
       if err_msg then
           print("(Reason: " .. err_msg .. ")") -- Optionally print the system error message
       end
       return false
   end
end


--=================================================

-- To Read Poscar
-- Assume the flos library is available in the path
local m = require "math"
local mc = require "flos.middleclass.middleclass"
local num = require "flos.num"

if not num then
  error("Failed to load Array from flos.num: 'array' field is nil")
end
local shape = require "flos.num.shape"

function ML_LBFGS:read_poscar_to_flos_array(args)

   subdirectory = (args.subdirectory and #args.subdirectory > 0) and args.subdirectory or "siesta_ml_1"
   local file, err = io.open(subdirectory .. "/" .. args.filename, "r")
  
  if not file then
    return nil, "Error opening file: " .. (err or "unknown error")
  end

  local lines = {}
  for line in file:lines() do
    table.insert(lines, line)
  end
  file:close()

  if #lines < 8 then
    return nil, "Error: File has fewer than 8 lines, likely invalid POSCAR format."
  end

  local function split_string_parts(str)
    local parts = {}
    for part in string.gmatch(str, "[^%s]+") do
      table.insert(parts, part)
    end
    return parts
  end

  local function parse_coord_line_to_table(str)
   local parts = {}
   for part in string.gmatch(str, "[^%s]+") do
     table.insert(parts, part)
   end
   if #parts < 3 then
     return nil, nil, "Coordinate line needs at least 3 values"
   end
   local coords = {}
   for i = 1, 3 do
     local num = tonumber(parts[i])
     if not num then
       return nil, nil, "Coordinate value is not a number: " .. parts[i]
     end
     table.insert(coords, num)
   end
   local symbol = parts[4] -- Atom symbol, if present
   return coords, symbol, nil
 end

 local scaling_factor = tonumber(lines[2])
  if not scaling_factor then
    return nil, "Error: Invalid scaling factor on line 2: " .. lines[2]
  end

  local temp_cell = {}
  for i = 3, 5 do
    local parts = split_string_parts(lines[i])
    if #parts < 3 then
      return nil, "Error: Lattice vector line " .. i .. " has fewer than 3 numbers: " .. lines[i]
    end
    local vector = {}
    for j = 1, 3 do
      local num = tonumber(parts[j])
      if not num then
        return nil, "Error: Non-numeric value in lattice vector line " .. i .. ": " .. parts[j]
      end
      table.insert(vector, num)
    end
    table.insert(temp_cell, vector)
  end
  local cell_array = num.Array.from(temp_cell)
  if cell_array.shape[1] ~= 3 or cell_array.shape[2] ~= 3 then
    return nil, "Error: Parsed cell data did not result in a 3x3 Array."
  end

  local atom_counts_str = split_string_parts(lines[7])
  local total_atoms = 0
  for i, count_str in ipairs(atom_counts_str) do
    local count = tonumber(count_str)
    if not count or count < 0 or count ~= math.floor(count) then
      return nil, "Error: Invalid atom count on line 7: " .. tostring(count_str)
    end
    total_atoms = total_atoms + count
  end

  local coord_type_line = 8
  if string.match(lines[coord_type_line]:lower(), "^%s*selective%s+dynamics%s*$") then
    coord_type_line = coord_type_line + 1
    if #lines <= coord_type_line then
      return nil, "Error: Missing coordinate type line after Selective dynamics"
    end
  end

  local coord_type = string.match(lines[coord_type_line], "^%s*(%S+)%s*")
  if not coord_type then
    return nil, "Error: Could not determine coordinate type on line " .. coord_type_line
  end

  local temp_coord = {}
  local symbols = {}
  local coord_start_line = coord_type_line + 1

  if #lines < coord_start_line + total_atoms - 1 then
    return nil, "Error: Not enough lines for " .. total_atoms .. " atoms."
  end

  for i = coord_start_line, coord_start_line + total_atoms - 1 do
   local coords, symbol, parse_err = parse_coord_line_to_table(lines[i])
   if parse_err then
     return nil, "Error parsing coordinate line " .. i .. ": " .. parse_err
   end
   table.insert(temp_coord, coords)
   table.insert(symbols, symbol or nil) -- Store nil if no symbol
 end

 local coord_array = num.Array.from(temp_coord)
 if coord_array.shape[1] ~= total_atoms or coord_array.shape[2] ~= 3 then
   return nil, "Error: Parsed coordinate data did not result in a " .. total_atoms .. "x3 Array."
 end

 return { cell = cell_array, coord = coord_array, symbols = symbols }
end


-- local test_filename = "POSCAR"

-- local status, result = pcall(read_poscar_to_flos_array({test_filename,), test_filename)

-- if not status then
--   print("Error reading POSCAR:", result)
-- else
--   local poscar_data = result
--   print("\n--- Lattice Vectors (cell - flos.num.array Array) ---")
--   print(poscar_data.cell)

--   print("\n--- Atomic Coordinates (coord - flos.num.array Array) ---")
--   print(poscar_data.coord)

--   print("\n--- Atom Symbols ---")
--   for i, symbol in ipairs(poscar_data.symbols) do
--     print("Atom " .. i .. ": " .. tostring(symbol))
--   end

--   print("\n--- Example Access ---")
--   if poscar_data.cell and poscar_data.cell.shape[1] >= 1 and poscar_data.cell.shape[2] >= 1 then
--     print(string.format("Cell vector 1, component 1: %.12f", poscar_data.cell[1][1]))
--   end
--   if poscar_data.coord and poscar_data.coord.shape[1] >= 1 and poscar_data.coord.shape[2] >= 1 then
--     print(string.format("Coord atom 1, component 1: %.12f", poscar_data.coord[1][1]))
--   end
--   if poscar_data.symbols and #poscar_data.symbols >= 1 then
--     print("Symbol atom 1: " .. tostring(poscar_data.symbols[1]))
--   end
--   if poscar_data.symbols and #poscar_data.symbols >= 2 then
--     print("Symbol atom 2: " .. tostring(poscar_data.symbols[2]))
--   end
-- end

return ML_LBFGS
