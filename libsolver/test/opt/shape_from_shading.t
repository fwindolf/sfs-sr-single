local W,H = Dim("W",0), Dim("H",1)

local T = Unknown("Theta", float3, {W, H}, 0) -- [z, zx, zy]

local M = Array("Mask", uint8, {W, H}, 1) -- Mask
local I = Array("Image", float3, {W, H}, 2) 
local A = Array("Albedo", float3, {W, H}, 3) -- Albedo

local L = {}
for i=0,3 do -- lighting model parameters
    L[i] = Param("L_" .. i .. "",float, 4 + i)
end

local fx = Param("fx", float, 8)
local fy = Param("fy", float, 9)
local cx = Param("cx", float, 10)
local cy = Param("cy", float, 11)

local posX, posY = Index(0), Index(1)
local eps = 1e-9


function Valid(x, y) 
    return and_(InBounds(x, y), greater(M(x, y), 0))
end

function Normals(x, y)
    local i = x + posX
    local j = y + posY

    local z  = T(x, y, 0)
    local zx = T(x, y, 1)
    local zy = T(x, y, 2)

    local n_x = fx * zx
    local n_y = fy * zy
    local n_z = -z - zx  *  (i - cx) - zy * (j - cy);
    
    local len = sqrt(n_x * n_x + n_y * n_y + n_z * n_z)
    local dz = Select(greater(len, eps), len, eps)
    
    return Vector(n_x / dz, n_y / dz, n_z / dz, dz)
end

function Shading(x, y)
	local normal = Normals(x, y)
	local n_x = normal[0]
	local n_y = normal[1]
	local n_z = normal[2]

    return L[0]* n_x + L[1] * n_y + L[2] * n_z + L[3]
end

function Image(x, y)
    local a_r = A(x, y, 0)
    local a_g = A(x, y, 1)
    local a_b = A(x, y, 2)
    local s = Shading(x, y)

	return Vector(s * a_r, s * a_g, s * a_b)
end

Energy(Select(Valid(0, 0), I(0, 0) - Image(0, 0), 0))
