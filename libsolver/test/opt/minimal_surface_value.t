local W,H = Dim("W",0), Dim("H",1)

local X = Unknown("Weight", float, {W, H}, 0) -- Mask
local T = Array("Theta", float3, {W, H}, 1) -- [z, zx, zy]
local M = Array("Mask", uint8, {W, H}, 2) -- Mask


local fx = Param("fx", float, 3)
local fy = Param("fy", float, 4)
local cx = Param("cx", float, 5)
local cy = Param("cy", float, 6)

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

function Surface(x, y)
    local z = T(x, y, 0)
    local normal = Normals(x, y)
    local dz = normal[3] 
    return Select(Valid(x, y), abs(z * dz) / (fx * fy), 0)
end

Energy(Surface(0, 0) - X(0, 0))
