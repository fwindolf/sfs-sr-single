local W,H = Dim("W",0), Dim("H",1)

-- Unknown and Input image
local T = Unknown("Theta", float3, {W, H}, 0) -- [z, zx, zy]

local M = Array("Mask", uint8, {W, H}, 1) 
local Z = Array("Depth", float, {W, H}, 2)

function Valid(x, y) 
    return and_(InBounds(x, y), greater(M(x, y, 0), 0))
end

function ValidZ(x, y) 
    return and_(InBounds(x, y), greater(Z(x, y, 0), 0))
end

-- Gradient Forward Differences in X 

local function GradientX(x, y)
    return Select(ValidZ(x + 1, y), Z(x + 1, y) - Z(x, y), 0)
end
GradientX = ComputedArray("GradientX", {W, H}, GradientX(0, 0))

-- Gradient Forward Differences in Y

local function GradientY(x, y)
    return Select(ValidZ(x, y + 1), Z(x, y + 1) - Z(x, y), 0)
end
GradientY = ComputedArray("GradientY", {W, H}, GradientY(0, 0))

-- Auxiliary Theta Forward Differences in X 

function ThetaZ(x, y)
    return Vector(Z(x, y), GradientX(x, y), GradientY(x, y))
end

Exclude(Not(Valid(0, 0)))

Energy(T(0, 0) - ThetaZ(0, 0))