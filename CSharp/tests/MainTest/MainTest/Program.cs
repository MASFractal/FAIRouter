using AI.DataStructs.Algebraic;
using FAI.Router;

BaseRoutedElement element1 = new BaseRoutedElement() { Name = "Sonnet 4.6"};
BaseRoutedElement element2 = new BaseRoutedElement() { Name = "Gemmini 2.5 Flash"};


var data = Env.GetTopK("Напиши эссе", new List<BaseRoutedElement>() { element1, element2});

Console.WriteLine(data);