local W,H = Dim("W",0), Dim("H",1)

-- Unknown and Input image
local T = Unknown("Theta", float3, {W, H}, 0) -- [z, zx, zy]

local M = Array("Mask", uint8, {W, H}, 1) -- Mask
local A = Array("Albedo", float3, {W, H}, 2) -- Albedo
local Z = Array("Depth", float, {W, H}, 3) -- Depth
local I = Array("Image", float3, {W, H}, 4) -- Input image
local U = Array("Dual", float3, {W, H}, 5) -- Dual Variable

local L = {}
for i=0,3 do -- lighting model parameters
    L[i] = Param("L_" .. i .. "",float, 6 + i)
end

local fx = Param("fx", float, 10)
local fy = Param("fy", float, 11)
local cx = Param("cx", float, 12)
local cy = Param("cy", float, 13)

local nu = Param("nu", float, 14)
local kappa = Param("kappa", float, 15)
local gamma = Param("gamma", float, 16)

local w_nu = sqrt(nu)
local w_kappa = sqrt(0.5 * kappa)
local w_gamma = sqrt(gamma)

local posX, posY = Index(0), Index(1)
local eps = 1e-9
local threshold = 1e-1

-----------------------------------------------------
-- Basic functions
-----------------------------------------------------

function Valid(x, y) 
    return and_(InBounds(x, y), greater(M(x, y), 0))
end

local function Continuous(x, y, nx, ny) 
    return less(abs(Z(x, y) - Z(nx, ny)), threshold)
end

local function ValidNeighbors(x, y)
    local valid = Valid(x, y)
    for nx,ny in Stencil { {1,0}, {-1,0}, {0,1}, {0,-1} } do
        valid = and_(valid, Continuous(x, y, x + nx, x + ny), Valid(x + nx, y + ny))
    end
    valid = and_(valid, InBoundsExpanded(0, 0, 1))
    return valid
end
local ValidArray = ComputedArray("ValidNeighbors", { W, H }, ValidNeighbors(0, 0))


-----------------------------------------------------
-- Shading
-----------------------------------------------------

-- Normals based on Theta

local function Normals(x, y)
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
    
    return Vector(n_x / dz, n_y / dz, n_z / dz, len)
end
Normals = ComputedArray("Normals", {W, H}, Normals(0, 0))

-- Shading by applying light to Normals

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

function ShapeFromShading(x, y)
    local valid = Valid(x, y)
    return Select(valid, I(x, y) - Image(x, y), 0)
end

-----------------------------------------------------
-- Smoothness
-----------------------------------------------------

-- Use the norm of unnormalized normals to determine the size of the surface
--  element (area of projection of pixel onto surface)

local function Vertex(x, y)
    local d = Z(x,y)
    local i = x + posX
    local j = y + posY    
    return Vector(((i-cx)/fx)*d, ((j-cy)/fy)*d, d)
end
local VertexArray = ComputedArray("Vertex", {W, H}, Vertex(0, 0))

-- Laplacian on back-projected 3D Points
local function Smoothness(x, y)
    local valid = eq(ValidArray(x, y), 1)
    local mag = 4.0 * VertexArray(x, y)
    for nx,ny in Stencil { {1,0}, {-1,0}, {0,1}, {0,-1} } do
        mag = mag - VertexArray(x + nx, y + ny)
    end
    return Select(valid, mag, 0)
end

-----------------------------------------------------
-- Auxiliary
-----------------------------------------------------

-- Gradient Forward Differences in X 

local function GradientX(x, y)
    return Select(Valid(x + 1, y), Z(x + 1, y) - Z(x, y), 0)
end
GradientX = ComputedArray("GradientX", {W, H}, GradientX(0, 0))

-- Gradient Forward Differences in Y

local function GradientY(x, y)
    return Select(Valid(x, y + 1), Z(x, y + 1) - Z(x, y), 0)
end
GradientY = ComputedArray("GradientY", {W, H}, GradientY(0, 0))

-- Auxiliary Theta Forward Differences in X 

function ThetaZ(x, y)
    return Vector(Z(x, y), GradientX(x, y), GradientY(x, y))
end

-- Difference between Auxiliary Theta and Optimized Theta, only
--  valid if the depths in gradients are defined

function Auxiliary(x, y)
    return Select(Valid(x, y), ThetaZ(x, y) - T(x, y) + U(x, y), 0)
end


-----------------------------------------------------
-- Energies
-----------------------------------------------------

Exclude(Not(Valid(0, 0)))
UsePreconditioner(true)

Energy(w_gamma * ShapeFromShading(0, 0))
Energy(w_nu * Smoothness(0, 0))
Energy(w_kappa * Auxiliary(0, 0))