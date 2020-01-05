local W,H = Dim("W",0), Dim("H",1)

-- Unknown and Input image
local X = Unknown("X", float, {W, H}, 0)
local A = Array("A", float, {W,H}, 1)
local M = Array("M", uint8, {W,H}, 2)

-----------------------------------------------------
-- Basic functions
-----------------------------------------------------

local function ValidA(x, y)
    return and_(greater(A(x, y), 0), InBounds(x, y))
end

local function ValidX(x, y)
    return and_(greater(X(x, y), 0), InBounds(x, y))
end

local function ValidMask(x, y)
    return and_(greater(M(x, y), 0), InBounds(x, y))
end

-----------------------------------------------------
-- Interpolated values 
-----------------------------------------------------

local function Norm(x, y)
    local sumA = 0
    local numA = 0
    local sumX = 0
    local numX = 0

    for i, j in Stencil { {-1, 0}, {1, 0}, {0, -1}, {0, 1} } do
        local px = x + i
        local py = y + j
        
        sumA = sumA + Select(ValidA(px, py), A(px, py), 0)
        sumX = sumX + Select(ValidX(px, py), X(px, py), 0)

        numA = numA + Select(ValidA(px, py), 1, 0)
        numX = numX + Select(ValidX(px, py), 1, 0)
    end

    for i, j in Stencil { {-1, -1}, {-1, 1}, {1, -1}, {1, 1} } do
        local px = x + i
        local py = y + j
        sumA = sumA + Select(ValidA(px, py), A(px, py) / sqrt(2), 0)
        sumX = sumX + Select(ValidX(px, py), X(px, py) / sqrt(2), 0)

        numA = numA + Select(ValidA(px, py), 1.0 / sqrt(2), 0)
        numX = numX + Select(ValidX(px, py), 1.0 / sqrt(2), 0)
    end

    for i, j in Stencil { {-2, -1}, {-2, 0}, {-2, 1},  
                          {-1, -2}, {0, -2}, {1, -2},
                          {2, -1}, {2, 0}, {2, 1},  
                          {-1, 2}, {0, 2}, {1, 2}} do
        local px = x + i
        local py = y + j
        sumA = sumA + Select(ValidA(px, py), A(px, py) / 2.0, 0)
        sumX = sumX + Select(ValidX(px, py), X(px, py) / 2.0, 0)

        numA = numA + Select(ValidA(px, py), 0.5, 0)
        numX = numX + Select(ValidX(px, py), 0.5, 0)
    end

    local normA = Select(greater(numA, 0), sumA / numA, 0)
    local normX = Select(greater(numX, 0), sumX / numX, 0)

    local norm = Select(greater(numA, 0), normA, normX)
    return Select(greater(norm, 0), norm - X(x, y), 0)
end
Norm = ComputedArray("Norm", {W, H}, Norm(0, 0))

-----------------------------------------------------
-- Energies
-----------------------------------------------------

Exclude(Not(ValidMask(0, 0)))

Energy(Select(ValidA(0, 0), A(0, 0) - X(0, 0), Norm(0, 0)))
