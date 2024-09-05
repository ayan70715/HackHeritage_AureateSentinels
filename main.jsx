// // import { createRoot } from 'react-dom/client'
// // import App from './App.jsx'
// // import background from '../assets/bg-img2.jpg'
// // import Navbar from './Navbar.jsx'
// // import LoginSignup from './LoginSignup.jsx'
// // import '../styles/index.css'

// // const resgister_signin = document.getElementsByClassName("tabs")[0];
// // console.log(resgister_signin)


// // createRoot(document.getElementsByTagName('body')[0]).render(


// //   <div className="all" style={{ backgroundImage: `url(${background})` }}>
// //     <Navbar />

// //     <App />

// //     {/* <LoginSignup /> */}

// //   </div>
  
// // )


// import { createRoot } from 'react-dom/client';
// import { useState } from 'react'; // Import useState from react
// import App from './App.jsx';
// import background from '../assets/bg-img2.jpg';
// import Navbar from './Navbar.jsx';
// import LoginSignup from './LoginSignup.jsx';
// import '../styles/index.css';

// function Main() {
//   const [showLoginSignup, setShowLoginSignup] = useState(false);

//   const handleLoginClick = () => {
//     setShowLoginSignup(true);
//   };

//   return (
//     <div className="all" style={{ backgroundImage: `url(${background})` }}>
//       <Navbar onLoginClick={handleLoginClick} />
//       {showLoginSignup ? (
//         <LoginSignup />
//       ) : (
//         <App />
//       )}
//     </div>
//   );
// }

// createRoot(document.getElementsByTagName('body')[0]).render(<Main />);









// import { createRoot } from 'react-dom/client';
// import { useState } from 'react';
// import App from './App.jsx';
// import background from '../assets/bg-img2.jpg';
// import Navbar from './Navbar.jsx';
// import LoginSignup from './LoginSignup.jsx';
// import '../styles/index.css';

// function Main() {
//   const [activeTab, setActiveTab] = useState('home');
//   const [showLoginSignup, setShowLoginSignup] = useState(false);

//   const handleTabClick = (tab) => {
//     setActiveTab(tab);
//     if (tab === 'home') {
//       setShowLoginSignup(false);
//     } else {
//       setShowLoginSignup(true);
//     }
//   };

//   return (
//     <div className="all" style={{ backgroundImage: `url(${background})` }}>
//       <Navbar onTabClick={handleTabClick} />
//       {activeTab === 'home' ? (
//         <App />
//       ) : (
//         <LoginSignup />
//       )}
//     </div>
//   );
// }

// createRoot(document.getElementsByTagName('body')[0]).render(<Main />);




// import { createRoot } from 'react-dom/client';
// import { useState } from 'react';
// import App from './App.jsx';
// import Work from './work.jsx'; // Import the Work component
// import background from '../assets/bg-img2.jpg';
// import Navbar from './Navbar.jsx';
// import LoginSignup from './LoginSignup.jsx';
// import '../styles/index.css';

// function Main() {
//   const [activeTab, setActiveTab] = useState('home');
//   const [showLoginSignup, setShowLoginSignup] = useState(false);
//   const [showWork, setShowWork] = useState(false); // New state variable

//   const handleTabClick = (tab) => {
//     setActiveTab(tab);
//     if (tab === 'home') {
//       setShowLoginSignup(false);
//     } else {
//       setShowLoginSignup(true);
//     }
//   };

//   const handleWorkClick = () => {
//     setShowWork(true); // Set showWork to true when the Work button is clicked
//   };

//   return (
//     <div className="all" style={{ backgroundImage: `url(${background})` }}>
//       <Navbar onTabClick={handleTabClick} onWorkClick={handleWorkClick} />
//       {showWork ? (
//         <Work />
//       ) : (
//         activeTab === 'home' ? (
//           <App />
//         ) : (
//           <LoginSignup />
//         )
//       )}
//     </div>
//   );
// }

// createRoot(document.getElementsByTagName('body')[0]).render(<Main />);



import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { useState } from 'react';
import App from './App.jsx';
import Work from './work.jsx';
import background from '../assets/bg-img2.jpg';
import Navbar from './Navbar.jsx';
import LoginSignup from './LoginSignup.jsx';
import '../styles/index.css';

function Main() {
  const [showLoginSignup, setShowLoginSignup] = useState(false);

  return (
    <Router>
      <div className="all" style={{ backgroundImage: `url(${background})` }}>
        <Navbar />
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/work" element={<Work />} />
          <Route path="/login" element={<LoginSignup />} />
          <Route path="/register" element={<LoginSignup />} />
        </Routes>
      </div>
    </Router>
  );
}

createRoot(document.getElementsByTagName('body')[0]).render(<Main />);