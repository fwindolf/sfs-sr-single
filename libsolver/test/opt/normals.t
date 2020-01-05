local W,H = Dim("W",0), Dim("H",1)

-- Unknown and Input image
local N = Unknown("Normals", float3, {W, H}, 0)
local M = Array("Mask", uint8, {W, H}, 1) 
local T = Array("Theta", float3, {W, H}, 2) -- [z, zx, zy]

local fx = Param("fx", float, 3)
local fy = Param("fy", float, 4)
local cx = Param("cx", float, 5)
local cy = Param("cy", float, 6)

local posX, posY = Index(0), Index(1)
local eps = 1e-9

function Valid(x, y) 
    return and_(InBounds(x, y), greater(M(x, y, 0), 0))
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
    
    return Vector(n_x / dz, n_y / dz, n_z / dz)
end

Exclude(Not(Valid(0, 0)))

Energy(Normals(0, 0) - N(0, 0))
