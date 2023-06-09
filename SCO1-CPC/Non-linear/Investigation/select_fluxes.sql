SELECT 
    (SELECT count() * 
        (SELECT power
        FROM wphoton) AS SUN
        FROM Surfaces,Photons
        WHERE Photons.surfaceID=Surfaces.id
                AND Surfaces.Path LIKE '%Light%'
                AND Photons.side = 1) AS SUN, 
        (SELECT count() * 
            (SELECT power
            FROM wphoton)
            FROM Surfaces,Photons
            WHERE Photons.surfaceID=Surfaces.id
                    AND Surfaces.Path LIKE '%Cyl_abs%'
                    AND Photons.side = 1) AS ABS, 
            (SELECT count() * 
                (SELECT power
                FROM wphoton)
                FROM Surfaces,Photons
                WHERE Photons.surfaceID=Surfaces.id
                        AND Surfaces.Path LIKE '%aux%'
                        AND Photons.side = 1) AS AUX;