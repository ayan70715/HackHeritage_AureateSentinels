// import '../styles/Navbar.css'

// const Navbar = ({ onTabClick, onWorkClick }) => {
//   return (
//     <div className="navbar">
//       <h1>Team Aureate Sentinels</h1>
//       <div className="tabs">
//         <a onClick={() => onTabClick('home')}>Home</a>
//         <a onClick={onWorkClick}>Work</a>
//         <a onClick={() => onTabClick('register')}>Register</a>
//         <a onClick={() => onTabClick('signin')}>Sign In</a>

//       </div>
//     </div>
//   );
// }

// export default Navbar;




// import React from 'react';
// import '../styles/Navbar.css';

// const Navbar = ({ onTabClick, onWorkClick }) => {
//   return (
//     <div className="navbar">
//       <h1>Team Aureate Sentinels</h1>
//       <div className="tabs">
//         <a onClick={() => onTabClick('home')}>Home</a>
//         <a onClick={onWorkClick}>Work</a>
//         <a onClick={() => onTabClick('register')}>Register</a>
//         <a onClick={() => onTabClick('signin')}>Sign In</a>
//       </div>
//     </div>
//   );
// };

// export default Navbar;


import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Navbar.css';

const Navbar = () => {
  return (
    <div className="navbar">
      <h1>Team Aureate Sentinels</h1>
      <div className="tabs">
        <Link to="/">Home</Link>
        <Link to="/work">Work</Link>
        <Link to="/register">Register</Link>
        <Link to="/login">Sign In</Link>
      </div>
    </div>
  );
};

export default Navbar;