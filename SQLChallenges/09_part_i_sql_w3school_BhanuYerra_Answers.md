# Challenge Set 9
## Part I: W3Schools SQL Lab 

*Introductory level SQL*

--

This challenge uses the [W3Schools SQL playground](http://www.w3schools.com/sql/trysql.asp?filename=trysql_select_all). Please add solutions to this markdown file and submit.

1. Which customers are from the UK?
A. Seven (7) customers are from UK with CustomerIDs: 4, 11, 16, 19, 38, 53 & 72
#--- SQL query follows
SELECT *
FROM [Customers]
WHERE Country = "UK";



2. What is the name of the customer who has the most orders?
A. Ernst Handel (CustomerID 20) has 10 orders
#--- SQL query follows
ELECT C.CustomerName, COUNT(O.OrderID)
FROM [Customers] as C JOIN [Orders] as O on C.CustomerID = O.CustomerID
GROUP BY C.CustomerName
ORDER BY COUNT(O.OrderID) DESC
LIMIT 1;

3. Which supplier has the highest average product price?
A.
SupplierID	SupplierName	AvgPrice	NoOfsProds
18	Aux joyeux ecclésiastiques	140.75	2
#--- SQL query follows
SELECT S.SupplierID, S.SupplierName, AVG(P.Price) AS AvgPrice, COUNT(ProductID) as NoOfsProds
FROM  [Products] as P Join [Suppliers] AS S ON P.SupplierID = S.SupplierID
GROUP BY S.SupplierID
ORDER BY AvgPrice DESC
LIMIT 1;


4. How many different countries are all the customers from? (*Hint:* consider [DISTINCT](http://www.w3schools.com/sql/sql_distinct.asp).)
A.
COUNT(DISTINCT Country)
21
#--- SQL query follows
SELECT COUNT(DISTINCT Country) FROM [Customers];


5. What category appears in the most orders?
A. 
Dairy Products
Freq	CategoryName
100	Dairy Products
93	Beverages
84	Confections
67	Seafood
50	Meat/Poultry
49	Condiments
42	Grains/Cereals
33	Produce
#---
SELECT COUNT(OD.OrderDetailID) AS Freq, Tble.CategoryName 
FROM
[OrderDetails] as OD JOIN
	(SELECT P.ProductID, P.ProductName, Cat.CategoryName AS CategoryName, Cat.Description
		FROM [Products] as P JOIN [Categories] as Cat ON P.CategoryID = Cat.CategoryID) AS Tble 
     ON OD.ProductID=Tble.ProductID


6. What was the total cost for each order?
A.
#--- SQL query follows
SELECT OD.OrderID, SUM(OD.Quantity*P.Price)
FROM [OrderDetails] AS OD JOIN [Products] AS P ON OD.ProductID=P.ProductID
GROUP BY OD.OrderID;

7. Which employee made the most sales (by total price)?
A.
Cost	EmployeeID	FirstName	LastName
105696.49999999999	4	Margaret	Peacock
#--- SQL query follows
SELECT SUM(OD.Quantity*P.Price) AS Cost, Emp.EmployeeID, Emp.FirstName, Emp.LastName
	FROM [OrderDetails] AS OD JOIN [Products] AS P ON OD.ProductID=P.ProductID 
    	JOIN [Orders] AS O ON OD.OrderID=O.OrderID
        JOIN [Employees] AS Emp ON O.EmployeeID = Emp.EmployeeID
GROUP BY Emp.EmployeeID
ORDER BY Cost DESC
LIMIT 1;


8. Which employees have BS degrees? (*Hint:* look at the [LIKE](http://www.w3schools.com/sql/sql_like.asp) operator.)
A.
EmployeeID	LastName	FirstName	BirthDate
3	Leverling	Janet	1963-08-30
5	Buchanan	Steven	1955-03-04
#--- SQL query follows
SELECT EmployeeID, LastName, FirstName, BirthDate
FROM [Employees]
WHERE Notes LIKE '%BS%';


9. Which supplier of three or more products has the highest average product price? (*Hint:* look at the [HAVING](http://www.w3schools.com/sql/sql_having.asp) operator.)
A.
SupplierID	SupplierName	Prods	AvgPrice
4	Tokyo Traders	3	46
#--- SQL query follows
SELECT S.SupplierID, S.SupplierName, COUNT(P.SupplierID) as Prods, AVG(P.Price) as AvgPrice
FROM [Products] as P JOIN [Suppliers] as S ON P.SupplierID = S.SupplierID
GROUP BY P.SupplieriD
HAVING Prods >= 3
ORDER BY AvgPrice DESC
LIMIT 1;
