local W,H = Dim("W",0), Dim("H",1)

-- Unknown and Input image

local Z = Unknown("Depth", float, {W, H}, 0) -- Depth

local M = Array("Mask", uint8, {W, H}, 1) -- Mask
local Z0 = Array("DepthLr", float, {W, H}, 2) -- Low resolution upsampled depth
local T = Array("Theta", float3, {W, H}, 3) -- [z, zx, zy]
local U = Array("Dual", float3, {W, H}, 4) -- Dual Variable

local mu = Param("mu", float, 5)
local kappa = Param("kappa", float, 6)

local w_mu = sqrt(mu)
local w_kappa = sqrt(0.5 * kappa)

-----------------------------------------------------
-- Basic functions
-----------------------------------------------------

function Valid(x, y) 
    return and_(InBounds(x, y), greater(M(x, y, 0), 0))
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
-- Fitting
-----------------------------------------------------

function Fitting(x, y)
    return Select(Valid(x, y), Z0(x, y) - Z(x, y), 0)
end

-----------------------------------------------------
-- Energies
-----------------------------------------------------
Exclude(Not(Valid(0, 0)))
UsePreconditioner(true)

Energy(w_mu * Fitting(0, 0))
Energy(w_kappa * Auxiliary(0, 0))
