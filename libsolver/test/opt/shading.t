local W,H = Dim("W",0), Dim("H",1)

local S = Unknown("Shading", float3, {W, H}, 0) 

local M = Array("Mask", uint8, {W, H}, 1) -- Mask
local N = Array("Normals", float3, {W, H}, 2) -- Normals


local L = {}
for i=0,3 do -- lighting model parameters
    L[i] = Param("L_" .. i .. "",float, 3 + i)
end

function Valid(x, y) 
    return and_(InBounds(x, y), greater(M(x, y), 0))
end

function Shading(x, y)
	local normal = N(x, y)
	local n_x = normal[0]
	local n_y = normal[1]
	local n_z = normal[2]

    return L[0]* n_x + L[1] * n_y + L[2] * n_z + L[3]
end

Energy(S(0, 0) - Shading(0, 0))
