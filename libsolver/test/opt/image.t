local W,H = Dim("W",0), Dim("H",1)

local I = Unknown("Image", float3, {W, H}, 0) 

local M = Array("Mask", uint8, {W, H}, 1) -- Mask
local S = Array("Shading", float3, {W, H}, 2) -- Shading
local A = Array("Albedo", float3, {W, H}, 3) -- Albedo


function Valid(x, y) 
    return and_(InBounds(x, y), greater(M(x, y), 0))
end

function Image(x, y)
    return A(x, y) * S(x, y)
end

Exclude(Not(Valid(0, 0)))

Energy(I(0, 0) - Image(0, 0))
