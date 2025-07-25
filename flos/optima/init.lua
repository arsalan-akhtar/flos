
--[[ 
Create a table with the default parameters of 
the Optima functions that are going to be inherited.
--]]

-- Create returning table
local ret = {}

-- Add the LBFGS optimization to the returned
-- optimization table.
ret.LBFGS = require "flos.optima.lbfgs"
ret.ML_LBFGS = require "flos.optima.ml_lbfgs"
ret.FIRE = require "flos.optima.fire"
ret.ML = require "flos.optima.ml"
ret.CG = require "flos.optima.cg"
ret.Line = require "flos.optima.line"
ret.Lattice = require "flos.optima.lattice"

return ret
