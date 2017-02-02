--[[ 
This module implements the FIRE algorithm
for minimization of a functional with an accompanying
gradient.
--]]

local m = require "math"
local mc = require "flos.middleclass.middleclass"
local optim = require "flos.optima.base"

-- Create the FIRE class (inheriting the Optimizer construct)
local FIRE = mc.class("FIRE", optim.Optimizer)

function FIRE:initialize(tbl)
   -- Wrapper which basically does nothing..

   -- All variables are defined as given in the FIRE
   -- paper.

   -- Initial time-step (in fs)
   self.dt_init = 1.0
   -- Maximum time-step
   self.dt_max = 10 * self.dt_init

   -- Increment and decrement values
   self.f_inc = 1.1
   self.f_dec = 0.5
   
   -- The decrement value for the alpha parameter
   self.f_alpha = 0.99

   -- Initial alpha parameter
   self.alpha_init = 0.1

   -- Number of consecutive iterations required with
   -- P = F. v > 0 before we increase the time-step
   self.N_min = 5
   -- Counter for number of P > 0
   -- When n_P_pos >= N_min we step the time-step
   self.n_P_pos = 0
   
   -- Currently reached iteration
   self.niter = 0

   -- Specify the maximum step of the variables
   self.max_dF = 0.1
   -- this is the convergence tolerance of the gradient
   self.tolerance = 0.02
   self.is_optimized = false

   -- Ensure we update the elements as passed
   -- by new(...)
   if type(tbl) == "table" then
      for k, v in pairs(tbl) do
	 if k == "mass" then
	    self:set_mass(v)
	 else
	    self[k] = v
	 end
      end
   end

   -- Initialize the variables
   self:reset()

end

-- Reset the algorithm
-- Basically all variables that
-- are set should be reset
function FIRE:reset()
   self.dt = self.dt_init
   self.alpha = self.alpha_init
   if self.mass == nil then
      self:set_mass()
   end
end

-- Initialiaze the velocities
function FIRE:set_velocity(V)
   -- Set the internal current velocity
   self.V = V:copy()
end

-- Update the masses, if nil, all masses will be the same
function FIRE:set_mass(mass)
   if mass == nil then
      -- Create fake mass with all same masses
      -- No need to duplicate data, we simply
      -- create a metatable (deferred lookup table with
      -- the same return value).
      self.mass = setmetatable({},
			       { __index = 
				    function(t,k)
				       return 1.
				    end,
			       })
   else
      self.mass = mass
   end
end

-- Function to return the current iteration count
function FIRE:iteration()
   return self.niter
end

-- Correct the step-size (change of optimization variable)
-- by asserting that the norm of each vector is below
-- a given threshold.
function FIRE:correct_dF(dF)

   -- Calculate the maximum norm
   local max_norm = self.norm1D(dF):max()

   -- Now normalize the displacement
   local norm = self.max_dF / max_norm
   if norm < 1. then
      return dF * norm
   else
      return dF
   end
   
end

-- Calculate the step for the FIRE algorithm,
-- eventually the gradient should be minimized
function FIRE:optimize(F, G)

   if self.V == nil then
      -- Force the content of a velocity
      self:set_velocity(F * 0.)
   end

   -- Calculate P quantity
   local P = self.flatdot(G, self.V)

   local V
   if P > 0. then

      -- Update velocity
      V = (1. - self.alpha) * self.V

      --[[
	 Here there are two choices:
	 1. Either the update of the velocity is, per coordinate, or
	 2. The updated velocity is globally adjusted.

	 From an empirical investigation the per coordinate has proven
	 more stable.
      --]]

      --[[
      -- This is the globally adjusted version:
      V = V + self.alpha * G / m.sqrt(self.flatdot(G, G)) *
	 m.sqrt(self.flatdot(self.V, self.V))
      --]]

      -- Per coordinate version:
      for i = 1, #V do
	 V[i] = V[i] + self.alpha * G[i] / G[i]:norm() * self.V[i]:norm()
      end

      if self.n_P_pos > self.N_min then
	 
	 -- We have had _many_ positive P and we may increase time-step
	 self.dt = m.min(self.dt * self.f_inc, self.dt_max)
	 self.alpha = self.alpha * self.f_alpha

      end

      -- Increment counter for positive P
      self.n_P_pos = self.n_P_pos + 1

   else
      -- We have a negative power, and thus we are climbing up, reset velocity
      V = self.V * 0.

      -- Decrease time-step and reset alpha
      self.dt = self.dt * self.f_dec
      self.alpha = self.alpha_init

      -- Reset counter for negative P
      self.n_P_pos = 0

   end

   -- Now perform a typical MD step
   local dF = self:MD(V, G)

   -- Update the weight of the algorithm
   self.weight = m.abs(self.flatdot(G, dF))
   
   -- Correct to the max displacement
   dF = self:correct_dF(dF)

   -- Update the internal velocity
   self:set_velocity(V + G * self.dt)
   
   -- Determine whether we have optimized the parameter/functional
   self:optimized(G)

   -- Calculate next step
   local newF
   if not self.is_optimized then
      newF = F + dF
   else
      newF = F:copy()
   end

   -- Update iteration counter
   self.niter = self.niter + 1
   
   return newF
end

-- Regular MD step by a given velocity, and force
-- This MD is an euler with mid-point correction as implemented in SIESTA
-- Then this returns a dF which is the step of the parameters F
function FIRE:MD(V, G)
   -- V == velocity
   -- G == gradient/force

   -- The equation is:
   --   dF = V(0) * dT / 2 + V(dT) * dT / 2
   --      = V(0) * dT / 2 + [V(0) + G*dT] * dT / 2
   --      = [V(0) + G*dT / 2] * dT
   return (V + G * (self.dt / 2)) * self.dt
end


-- Function to determine whether the
-- FIRE algorithm has converged
function FIRE:optimized(G)
   -- Check convergence
   local norm = self.norm1D(G):max()

   -- Determine whether the algorithm is complete.
   self.is_optimized = norm < self.tolerance
   
end

-- Print information regarding the FIRE algorithm
function FIRE:info()

   print("")
   print(("FIRE: dT initial / current / max:  %.4f / %.4f / %.4f fs"):format(self.dt_init, self.dt, self.dt_max))
   print(("FIRE: alpha initial / current:  %.4f /  %.4f"):format(self.alpha_init, self.alpha))
   print(("FIRE: # of positive G.V %d"):format(self.n_P_pos))
   print(("FIRE: Tolerance %.4f"):format(self.tolerance))
   print(("FIRE: Maximum change %.4f "):format(self.max_dF))
   print("")

end

return FIRE